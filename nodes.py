import logging

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
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

        return io.NodeOutput(patched, conditioning)


NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
}
