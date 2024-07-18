import matplotlib.pyplot as plt
import numpy as np


def show_signals(
    signals: np.ndarray, sample_time: float, save_path: str = None, show: bool = True
) -> None:
    """
    Show signal in plot. The shape of the signals is (n_samples, sample_length, n_channels).
    We divide the signals into n_channels subplots. Make at most 5 plots per row.

    This plot is not very useful if there are too many channels or samples.

    Args:
        signals (np.ndarray): The signals to plot.
        sample_time (float): The time between samples.
        save_path (str): The path to save the plot.
        show (bool): Whether to show the plot.
    """
    num_samples, sample_length, num_channels = signals.shape
    num_rows = num_channels // 5 + 1
    num_cols = min(num_channels, 5)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

    for i, ax in enumerate(axs.flat):
        if i >= num_channels:
            ax.axis("off")
        else:
            ax.set_title(f"Channel {i}")
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude")

            # scale the x-axis to sample time

            for j in range(num_samples):
                ax.plot(
                    np.arange(sample_length) * sample_time / 1e-9,
                    signals[j, :, i],
                    label=f"Sample {j}",
                )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def show_signal(
    signal: np.ndarray,
    sample_time: float,
    num_anodes: int,
    num_cathodes: int,
    num_drifts: int,
    save_path: str = None,
    show: bool = True,
) -> None:
    """Show a signal signal, with the anodes, cathodes, and drifts separated.

    Plots all the anodes together, all the cathodes together, and all the drifts together.

    Args:
        signal (np.ndarray): Single signal to plot.
        sample_time (float): The time between samples.
        num_anodes (int): Number of anodes in the sensor
        num_cathodes (int): Number of cathodes in the sensor
        num_drifts (int): Number of drifts in the sensor
        save_path (str, optional): The path to save the plot. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """

    assert (
        signal.shape[1] == num_anodes + num_cathodes + num_drifts
    ), "Signal shape does not match the number of channels"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 7.5))

    fig.suptitle("Signal with Anodes, Cathodes, and Drifts Separated")

    ax1.set_title("Anodes")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Amplitude")

    ax2.set_title("Cathodes")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Amplitude")

    ax3.set_title("Drifts")
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Amplitude")

    # divide the signal. It is always ordered Cathode -> Anode -> Drift
    cathode_signal = signal[:, :num_cathodes]
    anode_signal = signal[:, num_cathodes : num_cathodes + num_anodes]
    drift_signal = signal[:, num_cathodes + num_anodes :]

    for i, anode in enumerate(anode_signal.T):
        ax1.plot(
            np.arange(anode.shape[0]) * sample_time / 1e-9,
            anode,
            label=f"Anode {i+1}",
        )

    for i, cathode in enumerate(cathode_signal.T)
        ax2.plot(
            np.arange(cathode.shape[0]) * sample_time / 1e-9,
            cathode,
            label=f"Cathode {i+1}",
        )

    for i, drift in enumerate(drift_signal.T):
        ax3.plot(
            np.arange(drift.shape[0]) * sample_time / 1e-9,
            drift,
            label=f"Drift {i+1}",
        )

    ax1.legend()
    ax2.legend()
    ax3.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
