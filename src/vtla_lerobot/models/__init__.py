"""VTLA model components and LeRobot policy plugin exports."""

try:
    import lerobot as _lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only in a broken environment
    raise ImportError(
        "LeRobot is required by vtla_lerobot. Install the pinned third-party checkout with "
        "`pip install -e third_party/lerobot` or install the matching lerobot package."
    ) from exc

from .configuration_vtla import VTLAConfig
from .modeling_vtla import VTLAPolicy
from .processor_vtla import make_vtla_pre_post_processors

__all__ = ["VTLAConfig", "VTLAPolicy", "make_vtla_pre_post_processors"]
