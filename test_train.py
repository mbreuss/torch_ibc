import os

import hydra
import wandb
import torch
from omegaconf import DictConfig
from plot import eval
from visualize_models import eval, visualize_results
import omegaconf
from ibc.train_manager import TrainingManager
from ibc.ibc_agent import IBCAgent


@hydra.main(config_path="ibc/config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)

    train_manager = hydra.utils.instantiate(cfg.train_manager)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.configure_device('cpu')
    train_manager.train_agent(agent)

    train_coords, test_coords, pixel_errors = eval(train_manager, agent)

    plot_dir = 'assets'
    visualize_results(train_coords,
                      test_coords,
                      pixel_errors,
                      train_manager._data_loader["test"].dataset.resolution,
                      os.path.join(plot_dir, 'ibs_test.png'),
                      200,
                      140)

    print('done')


if __name__ == "__main__":
    main()
