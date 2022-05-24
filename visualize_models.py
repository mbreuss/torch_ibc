import dataclasses
import pathlib
from typing import Dict, Tuple

import dcargs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull
from tqdm.auto import tqdm

from ibc.train_manager import TrainingManager
from ibc.ibc_agent import IBCAgent


def eval(train_manager: TrainingManager, agent):
    dataset_test = train_manager.data_loader["test"].dataset
    dataset_train = train_manager.data_loader["train"].dataset
    total_mse = 0
    num_small_err = 0
    pixel_error = []
    
    for batch in tqdm(train_manager._data_loader["test"]):
        input, target = batch
        prediction = agent.predict(input).cpu().numpy()
        target = target.cpu().numpy()

        pred_unscaled = np.array(prediction)
        pred_unscaled += 1
        pred_unscaled /= 2
        pred_unscaled[:, 0] *= dataset_test.resolution[0] - 1
        pred_unscaled[:, 1] *= dataset_test.resolution[1] - 1

        target_unscaled = np.array(target)
        target_unscaled += 1
        target_unscaled /= 2
        target_unscaled[:, 0] *= dataset_test.resolution[0] - 1
        target_unscaled[:, 1] *= dataset_test.resolution[1] - 1

        diff = pred_unscaled - target_unscaled
        error = np.asarray(np.linalg.norm(diff, axis=1))
        num_small_err += len(error[error < 1.0])
        pixel_error.extend(error.tolist())
        total_mse += (diff ** 2).mean(axis=1).sum()

        total_test = len(dataset_test)
        average_mse = total_mse / total_test
        print(f"Test set MSE: {average_mse} ({num_small_err}/{total_test})")

        test_coords = dataset_test.coordinates
        train_coords = dataset_train.coordinates
        return train_coords, test_coords, np.asarray(pixel_error)


def visualize_results(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    errors: np.ndarray,
    resolution: Tuple[int, int],
    plot_path: pathlib.Path,
    dpi: int,
    threshold: float
) -> None:
    errors[errors >= threshold] = threshold
    colormap = plt.cm.Reds
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=threshold)
    plt.scatter(
        train_coords[:, 0],
        train_coords[:, 1],
        marker="x",
        c="black",
        zorder=2,
        alpha=0.5,
    )
    plt.scatter(
        test_coords[:, 0],
        test_coords[:, 1],
        c=errors,
        cmap=colormap,
        norm=normalize,
        zorder=1,
    )
    plt.colorbar()

    # Find index of predictions with less than 1 pixel error and color them in blue.
    idxs = errors < 1.0
    plt.scatter(
        test_coords[idxs, 0],
        test_coords[idxs, 1],
        marker="o",
        c="blue",
        zorder=1,
        alpha=1.0,
    )

    # Add convex hull of train set.
    if train_coords.shape[0] > 2:
        for simplex in ConvexHull(train_coords).simplices:
            plt.plot(
                train_coords[simplex, 0],
                train_coords[simplex, 1],
                "--",
                zorder=2,
                alpha=0.5,
                c="black",
            )

    plt.xlim(0 - 2, resolution[1] + 2)
    plt.ylim(0 - 2, resolution[0] + 2)

    plt.savefig(plot_path, format="png", dpi=dpi)
    plt.close()