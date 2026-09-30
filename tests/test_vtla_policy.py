from __future__ import annotations

import torch

from lerobot.configs import FeatureType, PolicyFeature
from vtla_lerobot.models import VTLAConfig, VTLAPolicy


IMAGE_NAMES = [
    "observation.images.front_camera",
    "observation.images.left_hand_camera",
    "observation.images.right_hand_camera",
    "observation.images.left_tactile_left_camera",
    "observation.images.left_tactile_right_camera",
    "observation.images.right_tactile_left_camera",
    "observation.images.right_tactile_right_camera",
]


def make_config() -> VTLAConfig:
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(28,)),
        **{
            name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
            for name in IMAGE_NAMES
        },
    }
    return VTLAConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        device="cpu",
        dtype="float32",
        chunk_size=4,
        n_action_steps=4,
        image_size=(64, 64),
        vision_backbone_weights=None,
        tactile_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        cross_attention_layers=1,
        n_vae_encoder_layers=1,
        temporal_ensemble_coeff=None,
    )


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    batch = {
        "observation.state": torch.randn(batch_size, 28),
        "action": torch.randn(batch_size, 4, 16),
        "action_is_pad": torch.tensor([[False, False, False, False], [False, False, True, True]]),
    }
    batch.update({name: torch.randn(batch_size, 3, 64, 64) for name in IMAGE_NAMES})
    return batch


def test_policy_supports_distinct_state_and_action_dimensions() -> None:
    policy = VTLAPolicy(make_config())
    batch = make_batch()
    policy.train()
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics.keys() >= {"l1_loss", "kld_loss", "contact_probability"}
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_policy_select_action_returns_one_unnormalized_shape() -> None:
    policy = VTLAPolicy(make_config())
    observation = make_batch()
    observation.pop("action")
    observation.pop("action_is_pad")
    action = policy.select_action(observation)
    assert action.shape == (2, 16)
    assert torch.isfinite(action).all()
