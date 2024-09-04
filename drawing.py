import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable

from pathlib import Path


class Draw:
    def __init__(self, output_dir: Path = Path("plots"), interactive: bool = False):
        self.output_dir = output_dir
        self.interactive = interactive
        self.cmap = ["green", "red", "blue", "orange", "purple", "brown"]

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def _save_fig(self, name: str) -> None:
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
        )
        if self.interactive:
            plt.show()
        plt.close()

    def plot_loss_history(
        self, log_dict: dict[str, npt.NDArray], name: str
    ):
        plt.plot(np.arange(1, len(log_dict['loss']) + 1), log_dict['loss'], label="Training")
        plt.plot(np.arange(1, len(log_dict['val_loss']) + 1), log_dict['val_loss'], label="Validation")
        if 'outlier_loss' in log_dict:
            plt.plot(np.arange(1, len(log_dict['outlier_loss']) + 1), log_dict['outlier_loss'], label="Outlier")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def imshow_multi(
        self, images: list[npt.NDArray], titles: list[str], name: str = 'imshow-multi'
    ):
        assert len(images) == len(titles), 'number of images and titles mismatch'
        n = len(images)
        fig, axs = plt.subplots(1, n)
        for i in range(n):
            axs[i].imshow(images[i])
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        self._save_fig(name)

    def plot_val_loss_vs_latent_size(
        self, latent_sizes, best_val_losses, name: str
    ):
        plt.plot(latent_sizes, best_val_losses)
        plt.xlabel("Size of latent space")
        plt.ylabel("MSE")
        self._save_fig(name)

    def plot_loss_vs_latent_size(
        self, latent_sizes, best_val_losses, outlier_losses, name: str
    ):
        plt.plot(latent_sizes, best_val_losses)
        plt.plot(latent_sizes, outlier_losses)
        plt.xlabel("Size of latent space")
        plt.ylabel("MSE")
        self._save_fig(name)

    def plot_loss_ratios(
        self, latent_sizes, loss_ratios, name: str
    ):
        plt.plot(latent_sizes, loss_ratios)
        plt.xlabel("Size of latent space")
        plt.ylabel("MSE(val) / MSE(outlier)")
        self._save_fig(name)

    def make_equivariance_plot(
        self,
        image: npt.NDArray,
        f: Callable[npt.NDArray, npt.NDArray],  # symmetry transformation
        g: Callable[npt.NDArray, npt.NDArray],  # mapping of the model
        name: str
    ):
        fig, axs = plt.subplots(2, 3)

        axs[0, 0].imshow(image)
        axs[0, 1].imshow(f(image))
        axs[0, 2].imshow(g(f(image)))
        axs[1, 0].imshow(image)
        axs[1, 1].imshow(g(image))
        axs[1, 2].imshow(f(g(image)))

        mse = round(float(np.mean((g(f(image)) - f(g(image)))**2)), 2)

        xmax, ymax = image.shape

        axs[0, 0].annotate('', xy=(1.35, 0.5), xycoords='axes fraction', 
                           xytext=(1, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 0].text(xmax+1, ymax/2-1, 'trans')

        axs[0, 1].annotate('', xy=(1.35, 0.5), xycoords='axes fraction', 
                           xytext=(1, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 1].text(xmax+1, ymax/2-1, 'pred')

        axs[1, 0].annotate('', xy=(1.35, 0.5), xycoords='axes fraction', 
                           xytext=(1, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 0].text(xmax+1, ymax/2-1, 'pred')

        axs[1, 1].annotate('', xy=(1.35, 0.5), xycoords='axes fraction', 
                           xytext=(1, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 1].text(xmax+1, ymax/2-1, 'trans')

        axs[0, 2].annotate('', xy=(0.5, -0.3), xycoords='axes fraction', 
                           xytext=(0.5, 0.), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='<->'))
        axs[0, 2].text(xmax/2+1, ymax+6, f'MSE = {mse}')

        for row in axs:
            for ax in row:
                ax.axis('off')
        
        plt.tight_layout()

        self._save_fig(name)
