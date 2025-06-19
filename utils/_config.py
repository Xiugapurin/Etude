from pathlib import Path
import json
from typing import Dict, Any


class CustomDict(dict):
    def __init__(self, config: Dict):
        super().__init__(config)

    def __getattr__(self, name: str) -> Dict | Any:
        value = self[name]
        if isinstance(value, dict):
            return CustomDict(value)
        else:
            return value

    def __getitem__(self, key: Any) -> Any:
        item = super().__getitem__(key)
        if isinstance(item, dict):
            return CustomDict(item)
        else:
            return item

import json
import os

class ConfigLoader:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON config: {e}")

    def __getitem__(self, key):
        return self._config[key]

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"'ConfigLoader' object has no attribute '{name}'")

    def get_config(self):
        return self._config



root = Path(__file__).parent.parent
path_config = root / "config.json"
with open(path_config, "r") as f:
    config_json = json.load(f)
config = CustomDict(config_json)
