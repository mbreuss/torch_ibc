import os

import hydra
import torch
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    dataset = hydra.utils.instantiate(cfg.dataset)
    bc_agent = hydra.utils.instantiate(cfg.agent)

