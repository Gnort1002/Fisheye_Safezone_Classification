import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


def plot_results(file="path/to/results.txt", dir=""):
    # Plot training results from .txt file.
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    ax = ax.ravel()

    try:
        with open(file, "r") as f:
            lines = f.readlines()

        train_epochs = []
        val_epochs = []
        train_losses = []
        train_mious = []
        val_losses = []
        val_mious = []

        for line in lines:
            if line.startswith("Epoch"):
                parts = line.split()
                epoch = int(parts[1])
                mode = str(parts[2])
                loss = float(parts[4])
                miou = float(parts[7])
                if mode == "Training:":
                    train_epochs.append(epoch)
                    train_losses.append(loss)
                    train_mious.append(miou)
                if mode == "Validation:":
                    val_epochs.append(epoch)
                    val_losses.append(loss)
                    val_mious.append(miou)
        # breakpoint()
        x_train = np.array(train_epochs)
        x_val = np.array(val_epochs)
        y_train_loss = np.array(train_losses)
        y_train_miou = np.array(train_mious)
        y_val_loss = np.array(val_losses)
        y_val_miou = np.array(val_mious)

        ax[0].plot(
            x_train,
            y_train_loss,
            marker=".",
            label="Train Loss",
            linewidth=2,
            markersize=8,
        )
        ax[0].plot(
            x_train,
            gaussian_filter1d(y_train_loss, sigma=3),
            ":",
            label="Smoothed",
            linewidth=2,
        )
        ax[0].set_title("Training Loss", fontsize=12)

        ax[1].plot(
            x_train,
            y_train_miou,
            marker=".",
            label="Train mIoU",
            linewidth=2,
            markersize=8,
        )
        ax[1].plot(
            x_train,
            gaussian_filter1d(y_train_miou, sigma=3),
            ":",
            label="Smoothed",
            linewidth=2,
        )
        ax[1].set_title("Training mIoU", fontsize=12)

        ax[2].plot(
            x_val,
            y_val_loss,
            marker=".",
            label="Validation Loss",
            linewidth=2,
            markersize=8,
        )
        ax[2].plot(
            x_val,
            gaussian_filter1d(y_val_loss, sigma=3),
            ":",
            label="Smoothed",
            linewidth=2,
        )
        ax[2].set_title("Validation Loss", fontsize=12)

        ax[3].plot(
            x_val,
            y_val_miou,
            marker=".",
            label="Validation mIoU",
            linewidth=2,
            markersize=8,
        )
        ax[3].plot(
            x_val,
            gaussian_filter1d(y_val_miou, sigma=3),
            ":",
            label="Smoothed",
            linewidth=2,
        )
        ax[3].set_title("Validation mIoU", fontsize=12)

        for a in ax:
            a.legend()

        fig.savefig(save_dir / "results.png", dpi=200)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Call the function with the path to your .txt file
    result_path = (
        "/media/gnort/HDD/Work_New/PheNet-Fisheye/Fisheye_Segment/save/pspnet_fisheye_darea/results.txt"
    )
    plot_results(result_path)
