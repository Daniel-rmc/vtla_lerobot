"""Checkpointed preprocessing and postprocessing pipelines for VTLA."""

from typing import Any

import torch

from lerobot.processor import (
    ImageCropResizeProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_vtla import VTLAConfig


def make_vtla_pre_post_processors(
    config: VTLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create serializable pipelines, including the training-time image resize."""
    steps = make_default_policy_processor_steps(
        config,
        dataset_stats,
        normalizer_device=config.device,
    )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            steps.to_device,
            ImageCropResizeProcessorStep(resize_size=tuple(config.image_size)),
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
