import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import art3d
import numpy as np

DETECTOR_OFFSET = 0.3
DRIFT_SIZE = 0.2
DRIFT_SPACE = 0.2
ANODE_SIZE = 0.2
ANODE_SPACE = 0.2
CATHODE_SIZE = 1.8
CATHODE_SPACE = 0.2
DISPLAY_HEIGHT = 0.1
PLOT_MARGIN = 0.5
FIGURE_SIZE = 10


def electron_hole_trace(
    electron_drift_path: np.ndarray,
    hole_drift_path: np.ndarray,
    detector_size: np.ndarray,
    num_drifts: int = -1,
    num_anodes: int = -1,
    num_cathodes: int = -1,
    step_size: float | np.ndarray = -1.0,
    unit: str = "steps",
    view: str = "xy",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    The shape of the electron and hole drift paths are expected to be (num_electrons, num_steps, 3).

    Args:
    """
    if unit not in ["steps", "mm"]:
        raise ValueError(f"Points unit {unit} is not valid. Expected 'steps' or 'mm'.")

    if unit == "steps":
        electron_drift_path = electron_drift_path * step_size
        hole_drift_path = hole_drift_path * step_size

    legend = []

    if view == "xy":
        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[0], detector_size[1]))
        )

        plt.title("Electron and Holde Drift Paths in Sensor")
        _display_xy_detector(ax, detector_size, num_drifts, num_anodes)

        for i in range(electron_drift_path.shape[0]):
            (electron_label,) = plt.plot(
                electron_drift_path[i, :, 0],
                electron_drift_path[i, :, 1],
                "orange",
                label="Electron Drift Path",
            )
            (hole_label,) = plt.plot(
                hole_drift_path[i, :, 0],
                hole_drift_path[i, :, 1],
                "purple",
                label="Hole Drift Path",
            )

        legend.extend([electron_label, hole_label])

    elif view == "xz":
        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[0], detector_size[2]))
        )

        plt.title("Electron and Holde Drift Paths in Sensor")
        _display_xz_detector(ax, detector_size, num_drifts, num_anodes, num_cathodes)

        for i in range(electron_drift_path.shape[0]):
            (electron_label,) = plt.plot(
                electron_drift_path[i, :, 0],
                electron_drift_path[i, :, 2],
                "orange",
                label="Electron Drift Path",
            )
            (hole_label,) = plt.plot(
                hole_drift_path[i, :, 0],
                hole_drift_path[i, :, 2],
                "purple",
                label="Hole Drift Path",
            )

        legend.extend([electron_label, hole_label])

    elif view == "yz":
        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[1], detector_size[2]))
        )

        plt.title("Electron and Holde Drift Paths in Sensor")
        _display_yz_detector(ax, detector_size, num_cathodes)

        for i in range(electron_drift_path.shape[0]):
            (electron_label,) = plt.plot(
                electron_drift_path[i, :, 1],
                electron_drift_path[i, :, 2],
                "orange",
                label="Electron Drift Path",
            )
            (hole_label,) = plt.plot(
                hole_drift_path[i, :, 1],
                hole_drift_path[i, :, 2],
                "purple",
                label="Hole Drift Path",
            )

        legend.extend([electron_label, hole_label])

    elif view == "all":
        # The position of all tries to make it as visual as possible. It is 2-1. The top left is the xz, the top right is the yz, and the bottom is the xy. They share axis to make it easier to see the position of the points.

        # Calculate the overall dimensions for the figure
        total_height = detector_size[2] + detector_size[1]
        total_width = detector_size[0] + detector_size[1]

        # Calculate the aspect _
        fig, axd = plt.subplot_mosaic(
            [["upper left", "upper right"], ["lower left", "lower right"]],
            figsize=(_get_width_height(total_width, total_height)),
            layout="constrained",
            height_ratios=[detector_size[2], detector_size[1]],
            width_ratios=[detector_size[0], detector_size[1]],
        )

        ax1 = axd["upper left"]
        ax2 = axd["upper right"]
        ax3 = axd["lower left"]
        ax4 = axd["lower right"]

        _display_xz_detector(ax1, detector_size, num_drifts, num_anodes, num_cathodes)
        _display_yz_detector(ax2, detector_size, num_cathodes)
        _display_xy_detector(ax3, detector_size, num_drifts, num_anodes)

        for i in range(electron_drift_path.shape[0]):
            (electron_label,) = ax1.plot(
                electron_drift_path[i, :, 0],
                electron_drift_path[i, :, 2],
                "orange",
                label="Electron Drift Path",
            )
            (hole_label,) = ax1.plot(
                hole_drift_path[i, :, 0],
                hole_drift_path[i, :, 2],
                "purple",
                label="Hole Drift Path",
            )
            ax2.plot(
                electron_drift_path[i, :, 1],
                electron_drift_path[i, :, 2],
                "orange",
                label="Electron Drift Path",
            )
            ax2.plot(
                hole_drift_path[i, :, 1],
                hole_drift_path[i, :, 2],
                "purple",
                label="Hole Drift Path",
            )
            ax3.plot(
                electron_drift_path[i, :, 0],
                electron_drift_path[i, :, 1],
                "orange",
                label="Electron Drift Path",
            )
            ax3.plot(
                hole_drift_path[i, :, 0],
                hole_drift_path[i, :, 1],
                "purple",
                label="Hole Drift Path",
            )

        legend.extend([electron_label, hole_label])

        # Hide the ax1 x axis and ax2 y axis and labels
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")

        # Hide the last axis
        ax4.axis("off")

        fig.suptitle("Electron and Hole Drift Paths in the Sensor")

    elif view == "3d":
        plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))

        ax = plt.axes(projection="3d")
        ax.set_box_aspect([detector_size[0], detector_size[1], detector_size[2]])

        plt.title("Electron and Hole Drift Paths in Sensor")

        _display_3d_detector(ax, detector_size, num_drifts, num_anodes, num_cathodes)

        for i in range(electron_drift_path.shape[0]):
            (electron_label,) = ax.plot(
                electron_drift_path[i, :, 0],
                electron_drift_path[i, :, 1],
                electron_drift_path[i, :, 2],
                "orange",
                label="Electron Drift Path",
            )
            (hole_label,) = ax.plot(
                hole_drift_path[i, :, 0],
                hole_drift_path[i, :, 1],
                hole_drift_path[i, :, 2],
                "purple",
                label="Hole Drift Path",
            )

        legend.extend([electron_label, hole_label])

    else:
        raise ValueError(
            f"View {view} is not a valid view. Expected 'xy', 'yz', 'xz', 'all' or '3d'."
        )

    # create the labels for anode, cathode, and drift strips
    legend.extend(
        [
            patches.Patch(color="blue", label="Drift Strips"),
            patches.Patch(color="red", label="Anode"),
            patches.Patch(color="green", label="Cathode"),
        ]
    )

    # add the legend to the existing plot, but dont overwrite the previous legend
    plt.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if show:
        plt.show()
    plt.close()


def show_points(
    points: np.ndarray,
    detector_size: np.ndarray,
    num_drifts: int = -1,
    num_anodes: int = -1,
    num_cathodes: int = -1,
    step_size: float | np.ndarray = -1.0,
    unit: str = "steps",
    view: str = "xy",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Show the points in a scatter plot in the Sensor. The shape of the points is expected to be (num_points, 3), and it must be ordered in x, y, z.

    The units of the points can either be "steps" or "mm". If the units are in "steps", the points will be multiplied by the step size of the detector model.

    Args:
        points (np.ndarray): The points to display in the sensor.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int): Number of drift strips.
        num_anodes (int): Number of anodes.
        num_cathodes (int): Number of cathodes.
        step_size (float | np.ndarray): The step size of the detector model. If it is an array, it must have the same shape as the points.
        unit (str): The unit of the points. It can be either "steps" or "mm".
        view (str): The view of the sensor. It can be either "xy", "yz", "xz", "all" or "3d".
        size (float): The size of the figure.
        save_path (str): The path to save the figure. If None, the figure will not be saved.
        show (bool): Whether to show the figure or not.
    """
    if unit not in ["steps", "mm"]:
        raise ValueError(f"Points unit {unit} is not valid. Expected 'steps' or 'mm'.")

    if unit == "steps":
        if isinstance(step_size, np.ndarray):
            if np.any(step_size == -1.0):
                raise ValueError("step_size cannot be -1.0 when unit is 'steps'.")
        elif step_size == -1.0:
            raise ValueError("step_size cannot be -1.0 when unit is 'steps'.")

        points = points * step_size

    legend = []

    if view == "xy":
        if num_drifts == -1 or num_anodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts and anodes to be specified."
            )

        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[0], detector_size[1]))
        )

        plt.title("Points in the Sensor")

        _display_xy_detector(ax, detector_size, num_drifts, num_anodes)

        scatter = ax.scatter(
            points[:, 0], points[:, 1], c="black", marker="x", s=25, label="Points"
        )
        legend.append(scatter)
    elif view == "xz":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )

        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[0], detector_size[2]))
        )

        plt.title("Points in the Sensor")

        _display_xz_detector(ax, detector_size, num_drifts, num_anodes, num_cathodes)

        scatter = ax.scatter(
            points[:, 0], points[:, 2], c="black", marker="x", s=25, label="Points"
        )
        legend.append(scatter)

    elif view == "yz":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )

        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[1], detector_size[2]))
        )

        plt.title("Points in the Sensor")

        _display_yz_detector(ax, detector_size, num_cathodes)

        scatter = ax.scatter(
            points[:, 1], points[:, 2], c="black", marker="x", s=25, label="Points"
        )
        legend.append(scatter)

    elif view == "all":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )

        # The position of all tries to make it as visual as possible. It is 2-1. The top left is the xz, the top right is the yz, and the bottom is the xy. They share axis to make it easier to see the position of the points.

        # Calculate the overall dimensions for the figure
        total_height = detector_size[2] + detector_size[1]  # z + y
        total_width = detector_size[0] + detector_size[1]  # x + y

        # Calculate the aspect _
        fig, axd = plt.subplot_mosaic(
            [["upper left", "upper right"], ["lower left", "lower right"]],
            figsize=(_get_width_height(total_width, total_height)),
            layout="constrained",
            height_ratios=[detector_size[2], detector_size[1]],
            width_ratios=[detector_size[0], detector_size[1]],
        )

        ax1 = axd["upper left"]
        ax2 = axd["upper right"]
        ax3 = axd["lower left"]
        ax4 = axd["lower right"]

        _display_xz_detector(ax1, detector_size, num_drifts, num_anodes, num_cathodes)
        scatter = ax1.scatter(
            points[:, 0], points[:, 2], c="black", marker="x", s=25, label="Points"
        )

        _display_yz_detector(ax2, detector_size, num_cathodes)
        ax2.scatter(
            points[:, 1], points[:, 2], c="black", marker="x", s=25, label="Points"
        )

        _display_xy_detector(ax3, detector_size, num_drifts, num_anodes)
        ax3.scatter(
            points[:, 0], points[:, 1], c="black", marker="x", s=25, label="Points"
        )

        # Hide the ax1 x axis and ax2 y axis and labels
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")

        # Hide the last axis
        ax4.axis("off")

        legend.append(scatter)

        fig.suptitle("Points in the Sensor")

    elif view == "3d":
        plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))

        ax = plt.axes(projection="3d")
        ax.set_box_aspect([detector_size[0], detector_size[1], detector_size[2]])

        plt.title("Points in the Sensor")

        _display_3d_detector(ax, detector_size, num_drifts, num_anodes, num_cathodes)

        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c="black",
            marker="x",
            s=25,
            label="Points",
        )
        legend.append(scatter)

    else:
        raise ValueError(
            f"View {view} is not a valid view. Expected 'xy', 'yz', 'xz', 'all' or '3d'."
        )

    # create the labels for anode, cathode, and drift strips
    legend.extend(
        [
            patches.Patch(color="blue", label="Drift Strips"),
            patches.Patch(color="red", label="Anode"),
            patches.Patch(color="green", label="Cathode"),
        ]
    )

    # plt.title("Points in the Sensor")
    # add the legend to the existing plot, but dont overwrite the previous legend
    plt.legend(handles=legend, loc="center left", bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if show:
        plt.show()
    plt.close()


def histogram(
    points: np.ndarray,
    detector_size: np.ndarray,
    num_drifts: int = -1,
    num_anodes: int = -1,
    num_cathodes: int = -1,
    bins: int | tuple[int, int, int] = 10,
    view: str = "xy",
    save_path: str = None,
    show: bool = True,
) -> None:
    """Show a histogram of the points in the sensor. The shape of the points is expected to be (num_points, 3), and it must be ordered in x, y, z.

    ..warning::
        For now, as it is very difficult to see anything, the 3D view is not implemented.

    Args:
        points (np.ndarray): The points to display in the sensor.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int, optional): Number of drift strips. Defaults to -1.
        num_anodes (int, optional): Number of anodes. Defaults to -1.
        num_cathodes (int, optional): Number of cathodes. Defaults to -1.
        bins (int | tuple[int, int, int], optional): The number of bins or a tuple with the number of bins for each axis. Defaults to 10.
        view (str, optional): The view of the sensor. It can be either "xy", "yz", "xz" or "all". Defaults to "xy".
        save_path (str, optional): The path to save the figure. If None, the figure will not be saved. Defaults to
        show (bool, optional): Whether to show the figure or not. Defaults to True.

    Raises:
        ValueError: If the view is not valid.
        ValueError: If the view requires the number of drifts, anodes, and cathodes to be specified.
    """
    display_detector(detector_size, num_drifts, num_anodes, num_cathodes, view=view)
    if isinstance(bins, int):
        bins = (bins, bins, bins)

    if view == "all":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )

        axes = plt.gcf().axes

        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ax1.hist2d(
            points[:, 0],
            points[:, 2],
            bins=(bins[0], bins[2]),
            range=((0, detector_size[0]), (0, detector_size[2])),
            alpha=0.5,
            cmap="hot",
        )

        ax2.hist2d(
            points[:, 1],
            points[:, 2],
            bins=(bins[1], bins[2]),
            range=((0, detector_size[1]), (0, detector_size[2])),
            alpha=0.5,
            cmap="hot",
        )

        ax3.hist2d(
            points[:, 0],
            points[:, 1],
            bins=(bins[0], bins[1]),
            range=((0, detector_size[0]), (0, detector_size[1])),
            alpha=0.5,
            cmap="hot",
        )

        # Put again the detector limits in the plot (hist2d changes them)
        ax1.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
        ax1.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)
        ax2.set_xlim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)
        ax2.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)
        ax3.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
        ax3.set_ylim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)

        plt.gcf().suptitle("Histogram of Points in the Sensor")

    elif view == "xy":
        if num_drifts == -1 or num_anodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts and anodes to be specified."
            )
        ax = plt.gca()

        ax.hist2d(
            points[:, 0],
            points[:, 1],
            range=((0, detector_size[0]), (0, detector_size[1])),
            bins=(bins[0], bins[1]),
            cmap="hot",
        )

        # Put again the detector limits in the plot (hist2d changes them)
        ax.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
        ax.set_ylim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)

        plt.title("Histogram of Points in the Sensor")

    elif view == "xz":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )
        ax = plt.gca()

        ax.hist2d(
            points[:, 0],
            points[:, 2],
            range=((0, detector_size[0]), (0, detector_size[2])),
            bins=(bins[0], bins[2]),
            cmap="hot",
        )

        # Put again the detector limits in the plot (hist2d changes them)
        ax.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
        ax.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)

        plt.title("Histogram of Points in the Sensor")

    elif view == "yz":
        if num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of cathodes to be specified."
            )
        ax = plt.gca()

        ax.hist2d(
            points[:, 1],
            points[:, 2],
            range=((0, detector_size[1]), (0, detector_size[2])),
            bins=(bins[1], bins[2]),
            cmap="hot",
        )

        # Put again the detector limits in the plot (hist2d changes them)
        ax.set_xlim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)
        ax.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)

        plt.title("Histogram of Points in the Sensor")

    else:
        raise ValueError(
            f"View {view} is not a valid view. Expected 'xy', 'yz', 'xz', 'all' or '3d'."
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if show:
        plt.show()

    plt.close()


def error_histogram(
    output_points: np.ndarray,
    target_points: np.ndarray,
    detector_size: np.ndarray,
    num_drifts: int = -1,
    num_anodes: int = -1,
    num_cathodes: int = -1,
    bins: int | tuple[int, int, int] = 10,
    view: str = "xy",
    save_path: str = None,
    show: bool = True,
) -> None:
    """Show the errors as a histogram in the sensor. The shape of the points is expected to be (num_points, 3), and it must be ordered in x, y, z.

    Args:
        output_points (np.ndarray): The output points to compare with the target.
        target_points (np.ndarray): The target points to compare with the output.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int, optional): Number of drift strips. Defaults to -1.
        num_anodes (int, optional): Number of anodes. Defaults to -1.
        num_cathodes (int, optional): Number of cathodes. Defaults to -1.
        bins (int | tuple[int, int, int], optional): The number of bins or a tuple with the number of bins for each axis. Defaults to 10.
        view (str, optional): The view of the sensor. It can be either "xy", "yz", "xz" or "all". Defaults to "xy".
        save_path (str, optional): The path to save the figure. If None, the figure will not be saved. Defaults to
        show (bool, optional): Whether to show the figure or not. Defaults to True.
    """

    display_detector(detector_size, num_drifts, num_anodes, num_cathodes, view=view)
    if isinstance(bins, int):
        bins = (bins, bins, bins)

    errors = np.linalg.norm(output_points - target_points, axis=1)

    # Create the histogram manually, as we need to plot the error in the sensor
    error_grid = np.zeros(bins)
    counts_grid = np.zeros(bins)

    x_bins = np.linspace(0, detector_size[0], bins[0] + 1)
    y_bins = np.linspace(0, detector_size[1], bins[1] + 1)
    z_bins = np.linspace(0, detector_size[2], bins[2] + 1)

    for i, target in enumerate(target_points):
        x = np.digitize(target[0], x_bins) - 1
        y = np.digitize(target[1], y_bins) - 1
        z = np.digitize(target[2], z_bins) - 1

        error_grid[x, y, z] += errors[i]
        counts_grid[x, y, z] += 1

    error_grid = np.divide(
        error_grid, counts_grid, out=np.zeros_like(error_grid), where=counts_grid != 0
    )

    if view == "all":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )

        axes = plt.gcf().axes

        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ax1.imshow(
            error_grid.sum(axis=1).T,
            extent=(0, detector_size[0], 0, detector_size[2]),
            origin="lower",
            cmap="hot",
            alpha=np.where(counts_grid.sum(axis=1).T == 0, 0, 0.75),
        )

        # Put again the detector limits in the plot (hist2d changes them)
        ax1.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
        ax1.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)
        ax2.set_xlim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)
        ax2.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)
        ax3.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
        ax3.set_ylim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)

        plt.gcf().suptitle("Histogram of Errors in the Sensor")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if show:
        plt.show()

    plt.close()


def display_detector(
    detector_size: np.ndarray,
    num_drifts: int = -1,
    num_anodes: int = -1,
    num_cathodes: int = -1,
    view: str = "xy",
    save_path: str = None,
) -> None:
    """
    Displays the sensor for a given instantiated detector model.

    .. warning::
        This function does not call to ``plt.show()``. It is up to the user to call it. This is because the user may want to plot multiple things in the same figure, and be because the function is used in other functions that may want to plot multiple things in the same figure.

    Args:
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int): Number of drift strips.
        num_anodes (int): Number of anodes.
        num_cathodes (int): Number of cathodes.
        view (str): The view of the sensor. It can be either "xy", "yz", "xz", "all" or "3d".
        size (float): The size of the figure.
        save_path (str): The path to save the figure. If None, the figure will not be saved.
    """
    if view == "xy":
        if num_drifts == -1 or num_anodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts and anodes to be specified."
            )
        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[0], detector_size[1]))
        )

        _display_xy_detector(ax, detector_size, num_drifts, num_anodes)

    elif view == "xz":
        if num_drifts == -1 or num_anodes == -1 or num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of drifts, anodes, and cathodes to be specified."
            )
        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[0], detector_size[2]))
        )

        _display_xz_detector(ax, detector_size, num_drifts, num_anodes, num_cathodes)

    elif view == "yz":
        if num_cathodes == -1:
            raise ValueError(
                f"The view {view} requires the number of cathodes to be specified."
            )
        _, ax = plt.subplots(
            figsize=(_get_width_height(detector_size[1], detector_size[2]))
        )

        _display_yz_detector(ax, detector_size, num_cathodes)

    elif view == "all":
        # The position of all tries to make it as visual as possible. It is 2-1. The top left is the xz, the top right is the yz, and the bottom is the xy. They share axis to make it easier to see the position of the points.

        # Calculate the overall dimensions for the figure
        total_height = detector_size[2] + detector_size[1]
        total_width = detector_size[0] + detector_size[1]

        _, axd = plt.subplot_mosaic(
            [["upper left", "upper right"], ["lower left", "lower right"]],
            figsize=(_get_width_height(total_width, total_height)),
            layout="constrained",
            height_ratios=[detector_size[2], detector_size[1]],
            width_ratios=[detector_size[0], detector_size[1]],
        )

        ax1 = axd["upper left"]
        ax2 = axd["upper right"]
        ax3 = axd["lower left"]
        ax4 = axd["lower right"]

        _display_xz_detector(ax1, detector_size, num_drifts, num_anodes, num_cathodes)
        _display_yz_detector(ax2, detector_size, num_cathodes)
        _display_xy_detector(ax3, detector_size, num_drifts, num_anodes)

        # Hide the ax1 x axis and ax2 y axis and labels
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")
        # Hide the last axis
        ax4.axis("off")

    elif view == "3d":
        plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))

        ax = plt.axes(projection="3d")
        ax.set_box_aspect([detector_size[0], detector_size[1], detector_size[2]])

        _display_3d_detector(ax, detector_size, num_drifts, num_anodes, num_cathodes)

    else:
        raise ValueError(
            f"View {view} is not a valid view. Expected 'xy', 'yz', 'xz', 'all' or '3d'."
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)


def _display_xy_detector(
    ax: plt.Axes,
    detector_size: np.ndarray,
    num_drifts: int,
    num_anodes: int,
) -> None:
    """Display the sensor in the xy plane.

    Args:
        ax (plt.Axes): plt.Axes object to plot the sensor.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int): Number of drift strips.
        num_anodes (int): Number of anodes.
    """
    # ax.set_aspect("equal")
    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("y-axis (mm)")

    # Plot the drift strips and the drift lines
    drift_line_pos = (3 * DRIFT_SIZE + 2 * DRIFT_SPACE) / 2
    drift_line_space = 3 * (DRIFT_SIZE + DRIFT_SPACE) + ANODE_SIZE + ANODE_SPACE

    for i in range(num_drifts):
        drift_x = DETECTOR_OFFSET + drift_line_pos + i * drift_line_space
        ax.plot(
            [drift_x, drift_x], [0, detector_size[1]], "--", color="black", linewidth=1
        )
        for j in range(3):
            ax.add_patch(
                patches.Rectangle(
                    (
                        DETECTOR_OFFSET
                        + i * drift_line_space
                        + j * (DRIFT_SIZE + DRIFT_SPACE),
                        0,
                    ),
                    DRIFT_SIZE,
                    -DISPLAY_HEIGHT,
                    fill=True,
                    color="blue",
                )
            )

    # Plot the anodes and label them
    anode_offset = DETECTOR_OFFSET + 3 * (DRIFT_SIZE + DRIFT_SPACE)
    anode_space = ANODE_SIZE + ANODE_SPACE + 3 * (DRIFT_SIZE + DRIFT_SPACE)

    for i in range(num_anodes):
        anode_x = anode_offset + i * anode_space
        ax.add_patch(
            patches.Rectangle(
                (anode_x, 0),
                ANODE_SIZE,
                -DISPLAY_HEIGHT,
                fill=True,
                color="red",
            )
        )
    # Plot the cathodes
    ax.add_patch(
        patches.Rectangle(
            (0, detector_size[1]),
            detector_size[0],
            DISPLAY_HEIGHT,
            fill=True,
            color="green",
        )
    )

    # Draw the border of the detector
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            detector_size[0],
            detector_size[1],
            fill=False,
            color="black",
        )
    )

    ax.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
    ax.set_ylim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)


def _display_xz_detector(
    ax: plt.Axes,
    detector_size: np.ndarray,
    num_drifts: int,
    num_anodes: int,
    num_cathodes: int,
) -> None:
    """Display the sensor in the xz plane.

    Args:
        ax (plt.Axes): plt.Axes object to plot the sensor.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int): Number of drift strips.
        num_anodes (int): Number of anodes.
        num_cathodes (int): Number of cathodes.
    """
    # ax.set_aspect("equal")
    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("z-axis (mm)")

    # Plot the drift strips and the drift lines
    drift_line_pos = (3 * DRIFT_SIZE + 2 * DRIFT_SPACE) / 2
    drift_line_space = 3 * (DRIFT_SIZE + DRIFT_SPACE) + ANODE_SIZE + ANODE_SPACE

    for i in range(num_drifts):
        drift_x = DETECTOR_OFFSET + drift_line_pos + i * drift_line_space
        ax.plot(
            [drift_x, drift_x], [0, detector_size[2]], "--", color="black", linewidth=1
        )
        for j in range(3):
            ax.add_patch(
                patches.Rectangle(
                    (
                        DETECTOR_OFFSET
                        + i * drift_line_space
                        + j * (DRIFT_SIZE + DRIFT_SPACE),
                        0,
                    ),
                    DRIFT_SIZE,
                    -DISPLAY_HEIGHT,
                    fill=True,
                    color="blue",
                )
            )

            ax.add_patch(
                patches.Rectangle(
                    (
                        DETECTOR_OFFSET
                        + i * drift_line_space
                        + j * (DRIFT_SIZE + DRIFT_SPACE),
                        detector_size[2],
                    ),
                    DRIFT_SIZE,
                    DISPLAY_HEIGHT,
                    fill=True,
                    color="blue",
                )
            )

    # Plot the anodes and label them
    anode_offset = DETECTOR_OFFSET + 3 * (DRIFT_SIZE + DRIFT_SPACE)
    anode_space = ANODE_SIZE + ANODE_SPACE + 3 * (DRIFT_SIZE + DRIFT_SPACE)

    for i in range(num_anodes):
        anode_x = anode_offset + i * anode_space
        ax.add_patch(
            patches.Rectangle(
                (anode_x, 0),
                ANODE_SIZE,
                -DISPLAY_HEIGHT,
                fill=True,
                color="red",
            )
        )

        ax.add_patch(
            patches.Rectangle(
                (anode_x, detector_size[2]),
                ANODE_SIZE,
                DISPLAY_HEIGHT,
                fill=True,
                color="red",
            )
        )

    # Plot the cathodes
    cathode_space = CATHODE_SIZE + CATHODE_SPACE

    for i in range(num_cathodes):
        cathode_y = i * cathode_space
        ax.add_patch(
            patches.Rectangle(
                (0, cathode_y),
                -DISPLAY_HEIGHT,
                CATHODE_SIZE,
                fill=True,
                color="green",
            )
        )

        ax.add_patch(
            patches.Rectangle(
                (detector_size[0], cathode_y),
                DISPLAY_HEIGHT,
                CATHODE_SIZE,
                fill=True,
                color="green",
            )
        )

    # Draw the border of the detector
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            detector_size[0],
            detector_size[2],
            fill=False,
            color="black",
        )
    )

    ax.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
    ax.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)


def _display_yz_detector(
    ax: plt.Axes,
    detector_size: np.ndarray,
    num_cathodes: int,
) -> None:
    """Display the sensor in the yz plane.

    Args:
        ax (plt.Axes): plt.Axes object to plot the sensor.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int): Number of drift strips.
        num_anodes (int): Number of anodes.
        num_cathodes (int): Number of cathodes.
    """
    # ax.set_aspect("equal")
    ax.set_xlabel("y-axis (mm)")
    ax.set_ylabel("z-axis (mm)")

    # Plot the cathodes
    cathode_space = CATHODE_SIZE + CATHODE_SPACE

    for i in range(num_cathodes):
        cathode_y = i * cathode_space
        ax.add_patch(
            patches.Rectangle(
                (detector_size[1], cathode_y),
                DISPLAY_HEIGHT,
                CATHODE_SIZE,
                fill=True,
                color="green",
            )
        )

    # plot the anodes. Even tho technically, should be a drift, an anode is more representative of the yz view
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            -DISPLAY_HEIGHT,
            detector_size[2],
            fill=True,
            color="red",
        )
    )

    # Draw the border of the detector
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            detector_size[1],
            detector_size[2],
            fill=False,
            color="black",
        )
    )

    ax.set_xlim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)
    ax.set_ylim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)


def _get_width_height(width: float, height: float) -> tuple[float, float]:
    """From an starting width and height, scale so all figures have the same aspect ratio, and same approximate size.

    It works by finding the smallest value of the 2, and then scaling both values by the biggest, and using the FIGURE_SIZE as the final size.

    Args:
        width (float): Width of the figure from the original size
        height (float): Height of the figure from the original

    Returns:
        tuple[float, float]: The new width and height
    """
    if width < height:
        new_height = FIGURE_SIZE
        new_width = width / height * new_height
    else:
        new_width = FIGURE_SIZE
        new_height = height / width * new_width

    return new_width, new_height


def _display_3d_detector(
    ax: plt.Axes,
    detector_size: np.ndarray,
    num_drifts: int,
    num_anodes: int,
    num_cathodes: int,
):
    """Display the sensor in 3D.

    Args:
        ax (plt.Axes): plt.Axes object to plot the sensor.
        detector_size (np.ndarray): The size of the detector in mm.
        num_drifts (int): Number of drift strips.
        num_anodes (int): Number of anodes.
        num_cathodes (int): Number of cathodes.
    """
    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("y-axis (mm)")
    ax.set_zlabel("z-axis (mm)")

    ax.set_xlim(-PLOT_MARGIN, detector_size[0] + PLOT_MARGIN)
    ax.set_ylim(-PLOT_MARGIN, detector_size[1] + PLOT_MARGIN)
    ax.set_zlim(-PLOT_MARGIN, detector_size[2] + PLOT_MARGIN)

    # 3d border with rectangles in patches
    rect = patches.Rectangle(
        (0, 0),
        detector_size[0],
        detector_size[1],
        fill=False,
        color="black",
    )
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")

    rect = patches.Rectangle(
        (0, 0),
        detector_size[0],
        detector_size[2],
        fill=False,
        color="black",
    )
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="y")

    rect = patches.Rectangle(
        (0, 0),
        detector_size[1],
        detector_size[2],
        fill=False,
        color="black",
    )
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="x")

    rect = patches.Rectangle(
        (0, 0),
        detector_size[0],
        detector_size[1],
        fill=False,
        color="black",
    )
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=detector_size[2], zdir="z")

    rect = patches.Rectangle(
        (0, 0),
        detector_size[0],
        detector_size[2],
        fill=False,
        color="black",
    )
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=detector_size[1], zdir="y")

    # Plot the drift strips and the drift lines
    drift_line_pos = (3 * DRIFT_SIZE + 2 * DRIFT_SPACE) / 2
    drift_line_space = 3 * (DRIFT_SIZE + DRIFT_SPACE) + ANODE_SIZE + ANODE_SPACE

    for i in range(num_drifts):
        drift_x = DETECTOR_OFFSET + drift_line_pos + i * drift_line_space
        ax.plot(
            [drift_x, drift_x],
            [0, detector_size[1]],
            [0, 0],
            "--",
            color="black",
            linewidth=1,
        )
        for j in range(3):
            rect = patches.Rectangle(
                (
                    DETECTOR_OFFSET
                    + i * drift_line_space
                    + j * (DRIFT_SIZE + DRIFT_SPACE),
                    0,
                    0,
                ),
                DRIFT_SIZE,
                detector_size[0],
                fill=True,
                color="blue",
                alpha=0.3,
            )
            ax.add_patch(rect)
            art3d.pathpatch_2d_to_3d(rect, z=0, zdir="y")

    # Plot the anodes and label them
    anode_offset = DETECTOR_OFFSET + 3 * (DRIFT_SIZE + DRIFT_SPACE)
    anode_space = ANODE_SIZE + ANODE_SPACE + 3 * (DRIFT_SIZE + DRIFT_SPACE)

    for i in range(num_anodes):
        anode_x = anode_offset + i * anode_space
        rect = patches.Rectangle(
            (anode_x, 0, 0),
            ANODE_SIZE,
            detector_size[0],
            fill=True,
            color="red",
            alpha=0.3,
        )
        ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=0, zdir="y")

    # Plot the cathodes
    cathode_space = CATHODE_SIZE + CATHODE_SPACE

    for i in range(num_cathodes):
        cathode_y = i * cathode_space
        rect = patches.Rectangle(
            (0, cathode_y, 0),
            detector_size[0],
            CATHODE_SIZE,
            fill=True,
            color="green",
            alpha=0.3,
        )
        ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=detector_size[1], zdir="y")
