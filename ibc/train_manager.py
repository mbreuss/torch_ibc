

import torch
import hydra
from omegaconf import DictConfig
import tqdm
import wandb


class TrainingManager:

    def __init__(self,
                 seed: int,
                 max_epochs: int,
                 train_batch_size: int,
                 test_batch_size: int,
                 num_workers: int,
                 dataset: DictConfig,
                 test_dataset: DictConfig):
        self._seed = seed
        self._max_epochs = max_epochs
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        self._num_workers = num_workers

        self._train_dataset = hydra.utils.instantiate(dataset)
        self._test_dataset = hydra.utils.instantiate(test_dataset)

        self._data_loader = self.make_dataloaders()

    def make_dataloaders(self):
        """
        Create a training and test dataloader using the dataset instances of the task
        """
        # check to avoid having coordinates in train and test set
        self._test_dataset.exclude(self._train_dataset.coordinates)

        train_dataloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_dataloader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return {
            "train": train_dataloader,
            "test": test_dataloader,
        }

    def train_agent(self, agent):

        for epoch in range(self._max_epochs):

            train_loss = []
            mean_mse = agent._evaluate(self._data_loader["test"])
            print('Epoch {}: Mean test mse is {}'.format(epoch, mean_mse))

            for batch in self._data_loader['train']:
                batch_loss = agent._train_step(*batch)
                train_loss.append(batch_loss)

            mean_train_loss = sum(train_loss) / len(train_loss)
            print('Mean train mse is {}'.format(mean_train_loss))
            #wandb.log({"loss": mean_train_loss, 'test_loss': mean_mse})
            # wandb.watch(agent._model)

        print('Training done!')


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    # wandb.config = cfg
    # wandb.init(project="EBM-Regression", entity="bennoq")
    train_manager = hydra.utils.instantiate(cfg.train_manager)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.configure_device('cuda')
    train_manager.train_agent(agent)
    print('done')


if __name__ == "__main__":
    main()