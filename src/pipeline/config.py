"""
Phase 4.1 — Configuration System.

Loads YAML configs, supports environment variable substitution,
and generates a config hash for reproducibility tracking.
"""
import os
import re
import hashlib
import yaml
from typing import Dict, Any


def _substitute_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with actual environment variable values."""
    pattern = r'\$\{(\w+)\}'
    def replacer(match):
        env_var = match.group(1)
        return os.environ.get(env_var, match.group(0))  # Keep original if not found
    return re.sub(pattern, replacer, str(value))


def _recursive_env_substitute(obj):
    """Recursively substitute env vars in all string values of a nested dict/list."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _recursive_env_substitute(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_env_substitute(item) for item in obj]
    return obj


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file with environment variable substitution.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        dict: The parsed configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = _recursive_env_substitute(config)
    
    return config


def config_hash(config: Dict[str, Any]) -> str:
    """
    Generate a SHA-256 hash of the config for reproducibility tracking.
    Two runs with the same config will produce the same hash.
    """
    config_str = yaml.dump(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
