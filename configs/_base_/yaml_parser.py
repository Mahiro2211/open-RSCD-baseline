import argparse
import yaml
import torch
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    train: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



