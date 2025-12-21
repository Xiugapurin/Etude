# etude/config/loader.py

"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .schema import EtudeConfig


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> EtudeConfig:
    """
    Load configuration with optional YAML overrides.

    Args:
        config_path: Path to YAML file with overrides (optional)
        overrides: Dict of programmatic overrides (optional)

    Returns:
        Complete EtudeConfig with all defaults filled in

    Example:
        # Use all defaults
        config = load_config()

        # Override from YAML
        config = load_config(Path("configs/default.yaml"))

        # Override programmatically
        config = load_config(overrides={"decoder": {"temperature": 0.8}})

        # Combine both
        config = load_config(
            Path("configs/default.yaml"),
            overrides={"env": {"device": "cuda"}}
        )
    """
    config_dict: Dict[str, Any] = {}

    # Load YAML overrides if provided
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                config_dict = _deep_merge(config_dict, yaml_config)

    # Apply programmatic overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Create config with validation
    return EtudeConfig(**config_dict)


def save_config(config: EtudeConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML.

    Useful for experiment tracking - saves the complete configuration
    used for a training run.

    Args:
        config: The configuration to save
        path: Output path for the YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
