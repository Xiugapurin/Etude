# etude/config/__init__.py

"""
Unified configuration system for Etude project.

Usage:
    from etude.config import load_config, EtudeConfig

    # Load with all defaults
    config = load_config()

    # Load with YAML overrides
    config = load_config(Path("configs/default.yaml"))

    # Access typed config values
    print(config.env.device)
    print(config.decoder.temperature)
"""

from .loader import load_config, save_config
from .schema import (
    EtudeConfig,
    EnvConfig,
    PathConfig,
    ExtractorConfig,
    BeatDetectorConfig,
    HFTConfig,
    DecoderConfig,
    PrepareConfig,
    TrainConfig,
    InferConfig,
    EvalConfig,
)

__all__ = [
    "load_config",
    "save_config",
    "EtudeConfig",
    "EnvConfig",
    "PathConfig",
    "ExtractorConfig",
    "BeatDetectorConfig",
    "HFTConfig",
    "DecoderConfig",
    "PrepareConfig",
    "TrainConfig",
    "InferConfig",
    "EvalConfig",
]
