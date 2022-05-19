from __future__ import annotations

import os
import sys

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class PolynomialSchedule:
  """Polynomial learning rate schedule for Langevin sampler."""

  def __init__(self, init, final, power, num_steps):
    self._init = init
    self._final = final
    self._power = power
    self._num_steps = num_steps

  def get_rate(self, index):
    """Get learning rate for index."""
    return ((self._init - self._final) *
            ((1 - (float(index) / float(self._num_steps-1))) ** (self._power))
            ) + self._final


class ExponentialSchedule:
  """Exponential learning rate schedule for Langevin sampler."""

  def __init__(self, init, decay):
    self._decay = decay
    self._latest_lr = init

  def get_rate(self, index):
    """Get learning rate. Assumes calling sequentially."""
    del index
    self._latest_lr *= self._decay
    return self._latest_lr


class DerivativeFreeOptimizer(nn.Module):

    def __init__(self,
                 noise_scale: float,
                 noise_shrink: float,
                 iters: int,
                 train_samples: int,
                 inference_samples: int
                 ):
        super(DerivativeFreeOptimizer, self).__init__()
        self._noise_scale = noise_scale
        self._noise_shrink = noise_shrink
        self._train_samples = train_samples
        self._iters = iters
        self._inference_samples = inference_samples
        self._bounds = None
        self._device = None

    def _get_device(self, device: torch.device):
        self._device = device

    def _get_bounds(self, bounds: np.ndarray):
        self._bounds = bounds

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self._bounds.shape[1])
        samples = np.random.uniform(self._bounds[0, :], self._bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self._train_samples)
        return samples.reshape(batch_size, self._train_samples, -1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Optimize for the best action given a trained EBM.
        """
        noise_scale = self._noise_scale
        bounds = torch.as_tensor(self._bounds).to(self._device)

        samples = self._sample(x.size(0) * self.inference_samples)
        samples = samples.reshape(x.size(0), self.inference_samples, -1)

        for i in range(self._iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]


class LangevinMCMCSampler(nn.Module):

    def __init__(self,
                 noise_scale: float,
                 noise_shrink: float,
                 iters: int,
                 train_samples: int,
                 inference_samples: int,
                 num_iterations: int =25,
                 sampler_stepsize_init: float = 1e-1,
                 sampler_stepsize_decay: float = 0.8, # if using exponential langevin rate.
                 sampler_stepsize_final: float = 1e-5,
                 sampler_stepsize_power: float = 2.0,
                 use_polynomial_rate: bool = True,  # default is exponential
                 delta_action_clip: int = 0.1):
        super(LangevinMCMCSampler, self).__init__()

        self._noise_scale = noise_scale
        self._noise_shrink = noise_shrink
        self._train_samples = train_samples
        self._iters = iters
        self._inference_samples = inference_samples
        self._delta_action_clip = delta_action_clip
        self._sampler_stepsize_init = sampler_stepsize_init
        self._sampler_stepsize_decay = sampler_stepsize_decay
        self._num_iterations = num_iterations
        self._gradient_scale = 0.5
        self._use_polynomial_rate = use_polynomial_rate,  # default is exponential
        self_sampler_stepsize_power = sampler_stepsize_power
        self._sampler_stepsize_final = sampler_stepsize_final
        self._bounds = None
        self._device = None

        # the Langevin MCMC uses a learning rate decay scheduler to adapt the step size of the sampling
        # based on the paper implementations
        if self._use_polynomial_rate:
            self._schedule = PolynomialSchedule(self._sampler_stepsize_init,
                                                self._sampler_stepsize_final,
                                                self_sampler_stepsize_power,
                                                self._num_iterations)
        else:
            self._schedule = ExponentialSchedule(self._sampler_stepsize_init,
                                                 self._sampler_stepsize_decay)

    def _get_device(self, device: torch.device):
        self._device = device

    def _get_bounds(self, bounds: np.ndarray):
        self._bounds = bounds

    def _sample(self,
                ebm: nn.Module,
                start_point,
                n_steps: int,
                step_size: int,
                return_intermediate_steps: bool = False) -> torch.Tensor:
        """
        Langevin Dynamics MCMC sampling method to draw samples from a distribution using the Langevin Dynamics

        :param:  num_samples:
        # TODO add description of stuff
        """
        delta_action_clip = self._delta_action_clip
        l_samples = []
        l_energies = []
        l_norms = []
        x = start_point
        for idx in range(n_steps):
            l_samples.append(start_point)
            e = ebm(x)
            grad = autograd.grad(e.sum(), x, only_inputs=True)[0]
            # scale gradients if actions would be in range of -1 to 1
            delta_action_clip = delta_action_clip * 0.5 * (self._bounds[1, :], self._bounds[1, :])
            # compute gradient norm
            grad_norm = grad.norm()
            # clip the gradients
            if self._grad_clip is not None:
                grad = torch.clip(grad, -self._grad_clip, self._grad_clip)
            noise = torch.randn_like(x) * self._noise_scale
            # this is in the Langevin dynamics equation.
            dynamics = step_size * (self._gradient_scale * grad + noise)
            x = x - dynamics
            x = torch.clip(x, self._bounds[1, :], self._bounds[1, :])
            l_energies.append(e.detach())
            l_norms.append(grad_norm.detach())

            # adapt the step size
            step_size = self._schedule.get_rate(idx + 1)

        if return_intermediate_steps:
            return l_samples, l_norms, l_energies
        else:
            return l_samples[-1], l_norms[-1], l_energies[-1]

    # TODO Fix batch adapted sampling method for MCMC langevin dynamics
    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        samples = self._sample(batch_size * self._train_samples)
        return samples.reshape(batch_size, self._train_samples, -1)

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Infer the best input for the conditional EBM given a current observation
        """
        pass
    # TODO add code here


@hydra.main(config_path="config", config_name="config.yaml")
def train_wrapper(cfg: DictConfig) -> None:

    dataset = CoordinateRegression(DatasetConfig(dataset_size=10))
    bounds = dataset.get_target_bounds()
    so = hydra.utils.instantiate(cfg.agent.StochasticOptimization)
    so._get_bounds(bounds)
    so._get_device('cpu')
    negatives = so.sample(64, nn.Identity())
    print(1)

if __name__ == "__main__":

    from dataset import CoordinateRegression, DatasetConfig
    train_wrapper()

