import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

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
