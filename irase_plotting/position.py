"""This file has functions related to plotting the positions in 3D coordinates in the i-RASE project. Such as histograms, scatter plots, but without plotting the sensor itself."""

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(
    points: np.ndarray,
    bins: int | tuple[int, int, int],
    view: str = "all",
    save_path: str = None,
    show: bool = True,
) -> None:
    """Plot a histogram of the points in 3D coordinates. If the view is set to "all", it will plot the histogram for each axis, otherwise it will plot only the histogram for the selected axis.

    Args:
        points (np.ndarray): The points must have shape (number of points, 3).
        bins (int | tuple[int, int, int]): The number of bins or a tuple with the number of bins for each axis.
        view (str, optional): The view of the histogram. Defaults to "all". Can be "all", "x", "y" or "z".
        save_path (str, optional): The path to save the plot. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Raises:
        ValueError: If the view is not "all", "x", "y" or "z".
        ValueError: If the view is "all" and the bins is not a tuple with 3 elements.
        ValueError: If the view is not "all" and the bins is not an integer.
    """

    if view not in ["all", "x", "y", "z"]:
        raise ValueError("The view must be 'all', 'x', 'y' or 'z'.")

    if view == "all" and (not isinstance(bins, tuple) or len(bins) != 3):
        raise ValueError(
            "If the view is 'all', the bins must be a tuple with the number of bins for each axis."
        )
    elif view != "all" and not isinstance(bins, int):
        raise ValueError("If the view is not 'all', the bins must be an integer.")

    if view == "all":
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].hist(points[:, 0], bins=bins[0], color="r")
        axs[0].set_title("X-axis")
        axs[0].set_xlabel("Position")
        axs[0].set_ylabel("Frequency")

        axs[1].hist(points[:, 1], bins=bins[1], color="g")
        axs[1].set_title("Y-axis")
        axs[1].set_xlabel("Position")

        axs[2].hist(points[:, 2], bins=bins[2], color="b")
        axs[2].set_title("Z-axis")
        axs[2].set_xlabel("Position")
        axs[2].set_ylabel("Frequency")

    else:
        axis = {"x": 0, "y": 1, "z": 2}[view]
        plt.hist(points[:, axis], bins=bins)
        plt.title(f"{view}-axis")
        plt.xlabel("Position")
        plt.ylabel("Frequency")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if show:
        plt.show()

    plt.close()
