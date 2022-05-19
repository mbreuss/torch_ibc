
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import DictConfig
import hydra


class BCAgent:

    def __init__(self,
                 model: DictConfig,
                 optimization: DictConfig):

        self._model = hydra.utils.instantiate(model)
        self._optimizer = hydra.utils.instantiate(optimization, params=self._model.parameters())
        self._device = None
        # TODO think about adding LR scheduler
        self.steps = 0

    def configure_device(self, device: torch.device):
        self._device = device

    def _train_step(self, input: torch.Tensor, target: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        self._model.train()

        input = input.to(self._device)
        target = target.to(self._device)

        out = self._model(input)
        loss = F.mse_loss(out, target)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        self.steps += 1
        return loss.item()

    @torch.no_grad()
    def _evaluate(self, dataloader: torch.utils.data.DataLoader):
        """
        Method for evaluating the model on one epoch of data
        """
        self._model.eval()

        total_mse = 0.0
        for input, target in tqdm(dataloader, leave=False):
            input = input.to(self._device)
            target = target.to(self._device)

            out = self._model(input)
            mse = F.mse_loss(out, target, reduction="none")
            total_mse += mse.mean(dim=-1).sum().item()

        mean_mse = total_mse / len(dataloader.dataset)
        return mean_mse

    @torch.no_grad()
    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self._model.eval()
        return self._model(input.to(self._device))

    def _get_eval_loss(self):
        NotImplementedError



@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    bc_agent = hydra.utils.instantiate(cfg.agent)
    print('done')
    print(1)


if __name__ == "__main__":
    main()