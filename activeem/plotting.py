from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(history: List[Dict], out: str | Path) -> None:
    df = pd.DataFrame(history)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df["n_train"], df["mae"], marker="o")
    plt.xlabel("Number of EM simulations")
    plt.ylabel("MAE")
    plt.title("ActiveEM learning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_bar_metrics(df: pd.DataFrame, metric: str, out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(df["method"], df[metric])
    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
