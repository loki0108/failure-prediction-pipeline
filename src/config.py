from pathlib import Path

import yaml

REQUIRED_KEYS = ["name", "raw_path", "db_path", "table", "target", "positive_label"]


def load_config(path):
    """Read a YAML config file and return it as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config {path} is missing keys: {missing}")

    return config