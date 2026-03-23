from pathlib import Path
import yaml


def load_config(config_path: str):

    path = Path(config_path)

    # If relative path, resolve from project root
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config