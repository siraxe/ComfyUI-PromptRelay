import logging

import torch
from comfy_api.latest import io

from .prompt_relay import (
    RelayConfig,
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .lora_schedule import (
    load_lora_weights,
    build_lora_assignment,
    build_blend_tensor,
    apply_lora_temporal_patches,
)

from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)


class PromptRelayEncode(io.ComfyNode):
    """Encodes temporal local prompts and patches the model for Prompt Relay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncode",
            display_name="Prompt Relay Encode",
            category="conditioning/prompt_relay",
            description=(
                "Encodes a global prompt combined with temporal local prompts and patches the model "
                "for Prompt Relay temporal control. Local prompts are separated by |. "
                "Use a standard CLIPTextEncode for the negative prompt."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Ordered prompts for each temporal segment, separated by |",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Comma-separated pixel space frame counts per segment. Leave empty to auto-distribute evenly.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Custom("PROMPT_RELAY_CONFIG").Output(display_name="relay_config"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon) -> io.NodeOutput:
        locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
        if not locals_list:
            raise ValueError("At least one local prompt is required (separate with |)")

        arch, patch_size, temporal_stride = detect_model_type(model)

        parsed_lengths = None
        if segment_lengths.strip():
            pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
            parsed_lengths = [max(1, round(p / temporal_stride)) for p in pixel_lengths]

        raw_tokenizer = get_raw_tokenizer(clip)
        full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

        log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
        for i, (s, e) in enumerate(token_ranges):
            log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

        samples = latent["samples"]
        latent_frames = samples.shape[2]
        tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

        effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

        log.info(
            "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
            latent_frames, tokens_per_frame, effective_lengths,
        )

        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
        mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

        patched = model.clone()
        apply_patches(patched, arch, mask_fn)

        config = RelayConfig(
            num_segments=len(locals_list),
            segment_lengths=effective_lengths,
            tokens_per_frame=tokens_per_frame,
            latent_frames=latent_frames,
            epsilon=epsilon,
            arch=arch,
            patch_size=patch_size,
        )

        return io.NodeOutput(patched, conditioning, config)


class PromptRelayLoraSchedule(io.ComfyNode):
    """Applies different LoRAs to different temporal segments defined by PromptRelayEncode."""

    @classmethod
    def define_schema(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return io.Schema(
            node_id="PromptRelayLoraSchedule",
            display_name="Prompt Relay LoRA Schedule",
            category="conditioning/prompt_relay",
            description=(
                "Applies different LoRAs to different temporal segments. Connect model and "
                "relay_config from PromptRelayEncode. Each LoRA slot maps directly to a segment "
                "(A→segment 0, B→segment 1, …). Set a slot to 'none' to skip that segment. "
                "extend_last controls whether the last assigned LoRA fills remaining segments."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Model output from PromptRelayEncode."),
                io.Custom("PROMPT_RELAY_CONFIG").Input("relay_config"),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="LoRA blend transition sharpness. Below ~0.1 gives sharp cuts (default 0.001). Use 0.5+ for softer crossfades.",
                ),
                io.Boolean.Input("extend_last", default=True, tooltip="If true, the last assigned LoRA fills any remaining segments beyond the LoRA slots."),
                io.Combo.Input("lora_name", options=["none"] + lora_list, tooltip="LoRA for segment 0."),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_name_2", options=["none"] + lora_list, optional=True, tooltip="LoRA for segment 1."),
                io.Float.Input("strength_2", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                io.Combo.Input("lora_name_3", options=["none"] + lora_list, optional=True, tooltip="LoRA for segment 2."),
                io.Float.Input("strength_3", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                io.Combo.Input("lora_name_4", options=["none"] + lora_list, optional=True, tooltip="LoRA for segment 3."),
                io.Float.Input("strength_4", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                io.Combo.Input("lora_name_5", options=["none"] + lora_list, optional=True, tooltip="LoRA for segment 4."),
                io.Float.Input("strength_5", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                io.Combo.Input("lora_name_6", options=["none"] + lora_list, optional=True, tooltip="LoRA for segment 5."),
                io.Float.Input("strength_6", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, relay_config, lora_name, strength,
                lora_name_2=None, strength_2=1.0,
                lora_name_3=None, strength_3=1.0,
                lora_name_4=None, strength_4=1.0,
                lora_name_5=None, strength_5=1.0,
                lora_name_6=None, strength_6=1.0,
                epsilon=1e-3, extend_last=True) -> io.NodeOutput:

        # Build slot map: each slot maps to a LoRA column index or None (no LoRA)
        all_slots = [
            (lora_name, strength),
            (lora_name_2, strength_2),
            (lora_name_3, strength_3),
            (lora_name_4, strength_4),
            (lora_name_5, strength_5),
            (lora_name_6, strength_6),
        ]

        lora_slot_map = []   # per-slot: LoRA column index or None
        lora_entries = []    # only the non-none entries in column order

        for name, s in all_slots:
            if name and name not in ("none", ""):
                lora_slot_map.append(len(lora_entries))
                lora_entries.append((name, s))
            else:
                lora_slot_map.append(None)

        if not lora_entries:
            log.info("[PromptRelay] LoRA Schedule: no LoRAs assigned, passing model through unchanged")
            return io.NodeOutput(model)

        num_loras = len(lora_entries)
        num_segments = relay_config.num_segments
        lora_assignment = build_lora_assignment(num_segments, lora_slot_map, extend_last)

        log.info(
            "[PromptRelay] LoRA Schedule: %d LoRAs for %d segments, assignment: %s",
            num_loras, num_segments, lora_assignment,
        )

        # Load all LoRA weights
        all_lora_deltas = []
        for name, s in lora_entries:
            deltas = load_lora_weights(name, model)
            # Apply strength to alpha_scale
            scaled = {k: (up, down, alpha * s) for k, (up, down, alpha) in deltas.items()}
            all_lora_deltas.append(scaled)

        # Build blend tensor [total_tokens, num_loras]
        blend = build_blend_tensor(
            relay_config.segment_lengths,
            relay_config.tokens_per_frame,
            lora_assignment,
            num_loras,
            device="cpu",
            epsilon=epsilon,
        )

        # Reorganize deltas by model key for patching
        # lora_deltas_by_key: {model_key: [(lora_up, lora_down, alpha_scale), ...]}
        lora_deltas_by_key = {}
        all_keys = set()
        for deltas in all_lora_deltas:
            all_keys.update(deltas.keys())

        for key in all_keys:
            per_lora = []
            for deltas in all_lora_deltas:
                if key in deltas:
                    per_lora.append(deltas[key])
                else:
                    per_lora.append(None)
            # Only include keys that have at least one LoRA
            if any(p is not None for p in per_lora):
                # Fill missing entries with zero-sized placeholders (no contribution)
                filled = []
                for p in per_lora:
                    if p is not None:
                        filled.append(p)
                    else:
                        # Create zero delta with same shape as a real one
                        ref = next(pp for pp in per_lora if pp is not None)
                        filled.append((
                            torch.zeros_like(ref[0]),
                            torch.zeros_like(ref[1]),
                            0.0,
                        ))
                lora_deltas_by_key[key] = filled

        # Patch model
        patched = model.clone()
        apply_lora_temporal_patches(patched, lora_deltas_by_key, blend)

        return io.NodeOutput(patched)


import folder_paths

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayLoraSchedule": PromptRelayLoraSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayLoraSchedule": "Prompt Relay LoRA Schedule",
}
