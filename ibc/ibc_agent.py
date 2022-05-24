import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import DictConfig
import hydra


class IBCAgent:

    def __init__(self,
                 model: DictConfig,
                 optimization: DictConfig,
                 stochastic_optimization: DictConfig,
                 lr_scheduler: DictConfig
                 ):

        self._model = hydra.utils.instantiate(model)
        self._optimizer = hydra.utils.instantiate(optimization, params=self._model.parameters())
        self._stochastic_optimizer = hydra.utils.instantiate(stochastic_optimization)
        self._lr_scheduler = hydra.utils.instantiate(lr_scheduler, optimizer=self._optimizer)
        self._device = None
        self.steps = 0

    def configure_device(self, device: torch.device):
        self._device = device
        self._stochastic_optimizer.get_device(device)
        self._model.get_device(device)

    def train_step(self, input: torch.Tensor, target: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        self._model.train()

        input = input.to(self._device)
        target = target.to(self._device)

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self._stochastic_optimizer.sample(input.size(0), self._model)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        print(input.shape)
        energy = self._model(input, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()
        self._lr_scheduler.step()

        self.steps += 1
        return loss

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        """
        Method for evaluating the model on one epoch of data
        """
        self._model.eval()

        total_mse = 0.0
        # for input, target in tqdm(dataloader, leave=False):
        for input, target in dataloader:
            input = input.to(self._device)
            target = target.to(self._device)

            out = self._stochastic_optimizer.infer(input, self._model)

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
        return self._stochastic_optimizer.infer(input.to(self._device), self._model)


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg)
    bc_agent = hydra.utils.instantiate(cfg.agent)
    print('done')
    print(1)


if __name__ == "__main__":
    main()