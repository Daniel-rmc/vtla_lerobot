"""LeRobot configuration for the VTLA policy."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE


DEFAULT_TACTILE_FEATURES = [
    "observation.images.left_tactile_left_camera",
    "observation.images.left_tactile_right_camera",
    "observation.images.right_tactile_left_camera",
    "observation.images.right_tactile_right_camera",
]


@PreTrainedConfig.register_subclass("vtla")
@dataclass
class VTLAConfig(PreTrainedConfig):
    """Configuration for the vision-tactile action-chunking policy.

    Feature dimensions are inferred by LeRobot from the dataset.  In particular,
    robot state and action dimensions are deliberately independent because the
    manipulationNet data uses a 28D state and a 16D Cartesian action.
    """

    chunk_size: int = 50
    n_action_steps: int = 1
    image_size: tuple[int, int] = (224, 224)
    tactile_feature_names: list[str] = field(default_factory=lambda: list(DEFAULT_TACTILE_FEATURES))
    dtype: str = "bfloat16"

    vision_backbone: str = "resnet18"
    tactile_backbone: str = "resnet18"
    vision_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    tactile_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    freeze_backbone_batch_norm: bool = True

    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 7
    cross_attention_layers: int = 2
    dropout: float = 0.1

    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    use_tactile_refine: bool = True
    tactile_refine_scale: float = 0.1
    temporal_ensemble_coeff: float | None = 0.01

    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError("VTLA currently supports exactly one observation timestep")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError("Temporal ensembling requires n_action_steps=1")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        if self.dim_model % self.n_heads != 0:
            raise ValueError("dim_model must be divisible by n_heads")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be one of: bfloat16, float16, float32")
        if not self.tactile_feature_names:
            raise ValueError("At least one tactile feature must be configured")
        if len(set(self.tactile_feature_names)) != len(self.tactile_feature_names):
            raise ValueError("tactile_feature_names contains duplicates")

    @property
    def tactile_features(self):
        if not self.input_features:
            return {}
        return {
            name: self.input_features[name]
            for name in self.tactile_feature_names
            if name in self.input_features
        }

    @property
    def vision_features(self):
        tactile_names = set(self.tactile_feature_names)
        return {name: feature for name, feature in self.image_features.items() if name not in tactile_names}

    def validate_features(self) -> None:
        if not self.input_features or OBS_STATE not in self.input_features:
            raise ValueError(f"VTLA requires the {OBS_STATE!r} feature")
        if self.input_features[OBS_STATE].type is not FeatureType.STATE:
            raise ValueError(f"{OBS_STATE!r} must be a STATE feature")
        if self.action_feature is None or not self.output_features or ACTION not in self.output_features:
            raise ValueError("VTLA requires the 'action' output feature")

        missing_tactile = sorted(set(self.tactile_feature_names) - set(self.image_features))
        if missing_tactile:
            raise ValueError(f"Configured tactile image features are missing: {missing_tactile}")
        if not self.vision_features:
            raise ValueError("VTLA requires at least one non-tactile image feature")

        image_shapes = {tuple(feature.shape) for feature in self.image_features.values()}
        if len(image_shapes) != 1:
            raise ValueError(f"All VTLA image features must share a shape, got {sorted(image_shapes)}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
