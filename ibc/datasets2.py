import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import hydra
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from omegaconf import DictConfig


class CoordinateRegression(Dataset):

    def __init__(self,
                 dataset_size: int,
                 resolution: list,
                 pixel_size: int,
                 pixel_color: list,
                 seed: int
                 ):
        if not pixel_size % 2:
            raise ValueError("'pixel_size' must be odd.")

        self.dataset_size = dataset_size
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.pixel_color = pixel_color
        self.seed = seed

        self.reset()

        self._coordinates = self._sample_coordinates(self.dataset_size)
        self._coordinates_scaled = self._scale_coordinates(self._coordinates)

    def reset(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)

    def exclude(self, coordinates: np.ndarray) -> None:
        """
        Exclude the given coordinates, if present, from the previously sampled ones.

        This is useful for ensuring the train set does not accidentally leak into the
        test set.
        """
        mask = (self.coordinates[:, None] == coordinates).all(-1).any(1)
        num_matches = mask.sum()
        while mask.sum() > 0:
            self._coordinates[mask] = self._sample_coordinates(mask.sum())
            mask = (self.coordinates[:, None] == coordinates).all(-1).any(1)
        self._coordinates_scaled = self._scale_coordinates(self._coordinates)
        print(f"Resampled {num_matches} data points.")

    def get_target_bounds(self) -> np.ndarray:
        """Return per-dimension target min/max."""
        return np.array([[-1.0, -1.0], [1.0, 1.0]])

    def _sample_coordinates(self, size: int) -> np.ndarray:
        """Helper method for generating pixel coordinates."""
        # Randomly generate pixel coordinates.
        u = np.random.randint(0, self.resolution[0], size=size)
        v = np.random.randint(0, self.resolution[1], size=size)

        # Ensure we remain within bounds when we take the pixel size into account.
        slack = self.pixel_size // 2
        u = np.clip(u, a_min=slack, a_max=self.resolution[0] - 1 - slack)
        v = np.clip(v, a_min=slack, a_max=self.resolution[1] - 1 - slack)

        return np.vstack([u, v]).astype(np.int16).T

    def _scale_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Helper method for scaling coordinates to the [-1, 1] range."""
        coords_scaled = np.array(coords, dtype=np.float32)
        coords_scaled[:, 0] /= self.resolution[0] - 1
        coords_scaled[:, 1] /= self.resolution[1] - 1
        coords_scaled *= 2
        coords_scaled -= 1
        return coords_scaled

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self.resolution + (3,)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def coordinates_scaled(self) -> np.ndarray:
        return self._coordinates_scaled

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        uv = self._coordinates[index]
        uv_scaled = self._coordinates_scaled[index]

        image = np.full(self.image_shape, fill_value=255, dtype=np.uint8)
        image[
            uv[0] - self.pixel_size // 2 : uv[0] + self.pixel_size // 2 + 1,
            uv[1] - self.pixel_size // 2 : uv[1] + self.pixel_size // 2 + 1,
        ] = self.pixel_color

        image_tensor = ToTensor()(image)
        target_tensor = torch.as_tensor(uv_scaled, dtype=torch.float32)

        return image_tensor, target_tensor


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    print(cfg)
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Visualize one instance.
    image, target = dataset[np.random.randint(len(dataset))]
    print(target)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()

    # Plot target distribution and convex hull.
    targets = dataset.coordinates
    plt.scatter(targets[:, 0], targets[:, 1], marker="x", c="black")
    for simplex in ConvexHull(targets).simplices:
        plt.plot(
            targets[simplex, 0],
            targets[simplex, 1],
            "--",
            zorder=2,
            alpha=0.5,
            c="black",
        )
    plt.xlim(0, dataset.resolution[1])
    plt.ylim(0, dataset.resolution[0])
    plt.show()

    # Plot target distribution and convex hull.
    targets = dataset.coordinates_scaled
    plt.scatter(targets[:, 0], targets[:, 1], marker="x", c="black")
    for simplex in ConvexHull(targets).simplices:
        plt.plot(
            targets[simplex, 0],
            targets[simplex, 1],
            "--",
            zorder=2,
            alpha=0.5,
            c="black",
        )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

    print(f"Target bounds:")
    print(dataset.get_target_bounds())
    print('done')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    main()

