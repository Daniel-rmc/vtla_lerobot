"""VTLA + LeRobot training package."""

__all__ = ["VTLAConfig", "VTLAPolicy", "make_vtla_pre_post_processors"]


def __getattr__(name: str):
    if name in __all__:
        from vtla_lerobot import models

        return getattr(models, name)
    raise AttributeError(name)
