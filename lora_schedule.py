import logging
import types

import folder_paths
import torch
import torch.nn.functional as F
from comfy.lora import model_lora_keys_unet
from comfy.lora_convert import convert_lora
from comfy.utils import load_torch_file

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA loading helpers
# ---------------------------------------------------------------------------

def load_lora_weights(lora_name, model):
    """Load a LoRA file and return {model_key: (lora_up, lora_down, alpha)}."""
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora_data = load_torch_file(lora_path, safe_load=True)
    lora_data = convert_lora(lora_data)

    diff_model = model.model
    key_map = model_lora_keys_unet(diff_model, {})

    from comfy.lora import load_lora, weight_adapter
    loaded = load_lora(lora_data, key_map, log_missing=False)

    deltas = {}
    for model_key, adapter in loaded.items():
        if adapter is None or not hasattr(adapter, "weights"):
            continue
        w = adapter.weights  # (lora_up, lora_down, alpha, mid, dora_scale, reshape)
        lora_up = w[0]
        lora_down = w[1]
        alpha = w[2]
        mid = w[3]
        dora_scale = w[4]

        if dora_scale is not None:
            log.warning("[PromptRelay] DoRA LoRAs not yet supported for temporal blending, skipping key: %s", model_key)
            continue

        if mid is not None:
            # LoCon: pre-compute combined down matrix
            # mid: [r, r2, k, k] style, reshape and combine
            mid_t = mid.transpose(0, 1).flatten(start_dim=1)
            down_t = lora_down.transpose(0, 1).flatten(start_dim=1)
            final_shape = [lora_down.shape[1], lora_down.shape[0], mid.shape[2], mid.shape[3]]
            lora_down = (
                torch.mm(down_t, mid_t)
                .reshape(final_shape)
                .transpose(0, 1)
            )

        alpha_scale = (alpha / lora_down.shape[0]) if alpha is not None else 1.0
        deltas[model_key] = (lora_up, lora_down, alpha_scale)

    if not deltas:
        raise ValueError(f"LoRA '{lora_name}' has no compatible weights for the current model")

    log.info("[PromptRelay] Loaded LoRA '%s': %d layer keys", lora_name, len(deltas))
    return deltas


def build_lora_assignment(num_segments, num_loras):
    """Map each segment index to a LoRA index (repeat last LoRA if fewer than segments)."""
    if num_loras == 0:
        raise ValueError("At least one LoRA is required")
    return [min(i, num_loras - 1) for i in range(num_segments)]


def build_blend_tensor(segment_lengths, tokens_per_frame, lora_assignment, num_loras, device):
    """Build a [total_tokens, num_loras] one-hot blend tensor.

    Each token is assigned to the LoRA of its segment.
    """
    total_tokens = sum(segment_lengths) * tokens_per_frame
    blend = torch.zeros(total_tokens, num_loras, device=device)

    frame_cursor = 0
    for seg_idx, seg_len in enumerate(segment_lengths):
        lora_idx = lora_assignment[seg_idx]
        token_start = frame_cursor * tokens_per_frame
        token_end = (frame_cursor + seg_len) * tokens_per_frame
        blend[token_start:token_end, lora_idx] = 1.0
        frame_cursor += seg_len

    return blend


# ---------------------------------------------------------------------------
# Layer patching
# ---------------------------------------------------------------------------

def _make_patched_linear(orig_module, lora_deltas_for_layer, blend_tensor):
    """Return a replacement forward() that applies temporal LoRA blending.

    lora_deltas_for_layer: list of (lora_up, lora_down, alpha_scale) per LoRA,
                           in the order they appear in blend_tensor columns.
    blend_tensor: [total_tokens, num_loras] on cpu (will be sliced per-batch).
    """

    def patched_forward(self_module, x):
        import comfy.model_management
        device = x.device
        dtype = x.dtype
        weight = comfy.model_management.cast_to_device(self_module.weight, device, dtype)
        bias = comfy.model_management.cast_to_device(self_module.bias, device, dtype) if self_module.bias is not None else None
        out = F.linear(x, weight, bias)
        num_tokens = x.shape[1]

        # Slice blend to match actual token count (may differ at boundary)
        if num_tokens <= blend_tensor.shape[0]:
            b = blend_tensor[:num_tokens].to(device=x.device, dtype=x.dtype)
        else:
            # Pad if somehow more tokens (shouldn't happen normally)
            b = F.pad(blend_tensor, (0, 0, 0, num_tokens - blend_tensor.shape[0])).to(
                device=x.device, dtype=x.dtype
            )

        for i, (lora_up, lora_down, alpha_scale) in enumerate(lora_deltas_for_layer):
            blend_col = b[:, i:i + 1]  # [tokens, 1]
            down_w = comfy.model_management.cast_to_device(lora_down, device, dtype)  # [rank, in_dim]
            up_w = comfy.model_management.cast_to_device(lora_up, device, dtype)  # [out_dim, rank]
            mid = x @ down_w.T  # [B, tokens, in_dim] @ [in_dim, rank] -> [B, tokens, rank]
            delta = mid @ up_w.T  # [B, tokens, rank] @ [rank, out_dim] -> [B, tokens, out_dim]
            out = out + (alpha_scale * blend_col) * delta

        return out

    return patched_forward


class _LinearPatch:
    """Descriptor that binds a patched forward onto a linear module."""

    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def __get__(self, obj, objtype=None):
        fn = self.forward_fn
        return types.MethodType(fn, obj)


def apply_lora_temporal_patches(model_clone, lora_deltas_by_key, blend_tensor):
    """Patch every linear layer that has LoRA deltas with temporal blending.

    lora_deltas_by_key: {model_key: [(lora_up, lora_down, alpha_scale), ...]}
                        One entry per LoRA, for each model key.
    blend_tensor: [total_tokens, num_loras] blend weights.
    """
    # Collect unique module keys (strip .weight suffix)
    module_keys = set()
    for model_key in lora_deltas_by_key:
        for suffix in (".weight", ".bias"):
            if model_key.endswith(suffix):
                module_keys.add(model_key[: -len(suffix)])
                break
        else:
            module_keys.add(model_key)

    patched = 0
    for mod_key in module_keys:
        # Build per-layer LoRA list: [(lora_up, lora_down, alpha_scale), ...]
        weight_key = mod_key + ".weight" if mod_key + ".weight" in lora_deltas_by_key else mod_key
        per_lora = lora_deltas_by_key.get(weight_key, [])
        if not per_lora:
            continue

        patch_key = f"{mod_key}.forward"
        if patch_key in getattr(model_clone, "object_patches", {}):
            log.warning("[PromptRelay] Skipping already-patched key: %s", patch_key)
            continue

        # Get the actual module to inspect its shape
        try:
            module = model_clone.model
            for attr in mod_key.split("."):
                module = getattr(module, attr)
        except AttributeError:
            log.warning("[PromptRelay] Cannot resolve module for key: %s", mod_key)
            continue

        patched_fn = _make_patched_linear(module, per_lora, blend_tensor)
        model_clone.add_object_patch(patch_key, _LinearPatch(patched_fn).__get__(module, type(module)))
        patched += 1

    log.info("[PromptRelay] Patched %d linear layers for temporal LoRA blending", patched)
