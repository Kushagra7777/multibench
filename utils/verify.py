"""Shape validation utilities for MultiBench."""
from typing import List

import torch
from torch import nn


def validate_shapes(
    encoders: List[nn.Module],
    fusion: nn.Module,
    head: nn.Module,
    sample_inputs: List[torch.Tensor],
) -> None:
    """Dry-run a forward pass to catch dimension mismatches at model-build time.

    Passes ``sample_inputs`` through each encoder, then through fusion, then
    through head, printing the tensor shape at each stage.  The entire pass
    runs under :func:`torch.inference_mode` with models temporarily switched
    to ``eval()`` mode, so it is safe to call before training starts.

    Args:
        encoders: List of encoder modules (one per modality), in the order
            matching ``sample_inputs``.
        fusion: Fusion module that combines encoder outputs.
        head: Prediction head that takes the fused representation.
        sample_inputs: List of tensors matching the expected input shapes (one
            per modality).  A batch dimension of 1 is recommended.

    Raises:
        ValueError: If the number of encoders and sample inputs don't match.
        RuntimeError: If any forward pass fails due to a shape mismatch,
            with a message indicating which stage failed.
    """
    if len(encoders) != len(sample_inputs):
        raise ValueError(
            f"Number of encoders ({len(encoders)}) must match number of "
            f"sample inputs ({len(sample_inputs)})"
        )

    train_states = [m.training for m in encoders]
    was_fusion_training = fusion.training
    was_head_training = head.training
    for enc in encoders:
        enc.eval()
    fusion.eval()
    head.eval()

    try:
        with torch.inference_mode():
            print("=== validate_shapes ===")
            encoder_outs = []
            for i, (enc, inp) in enumerate(zip(encoders, sample_inputs)):
                try:
                    out = enc(inp)
                    shape = (
                        tuple(out.shape)
                        if isinstance(out, torch.Tensor)
                        else [tuple(x.shape) for x in out]
                    )
                    print(f"  encoder[{i}]: {tuple(inp.shape)} -> {shape}")
                    encoder_outs.append(out)
                except Exception as exc:
                    raise RuntimeError(
                        f"encoder[{i}] failed with input shape {tuple(inp.shape)}: {exc}"
                    ) from exc

            try:
                fused = fusion(encoder_outs)
                fused_t = fused[0] if isinstance(fused, (tuple, list)) else fused
                in_shapes = [
                    tuple(x.shape) if isinstance(x, torch.Tensor) else x
                    for x in encoder_outs
                ]
                print(f"  fusion:      {in_shapes} -> {tuple(fused_t.shape)}")
            except Exception as exc:
                raise RuntimeError(f"fusion failed: {exc}") from exc

            try:
                result = head(fused_t)
                print(f"  head:        {tuple(fused_t.shape)} -> {tuple(result.shape)}")
            except Exception as exc:
                raise RuntimeError(
                    f"head failed with input shape {tuple(fused_t.shape)}: {exc}"
                ) from exc

            print("=== validate_shapes passed ===")
    finally:
        for enc, was_training in zip(encoders, train_states):
            enc.train(was_training)
        fusion.train(was_fusion_training)
        head.train(was_head_training)
