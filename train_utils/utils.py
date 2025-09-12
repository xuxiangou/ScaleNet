from typing import List
import pandas as pd
import numpy as np
import torch
import time
import os
from matplotlib import pyplot as plt


def create_dir(dir):
    fold_num = sum([os.path.isdir(dir + "/" + listx) for listx in os.listdir(dir)])
    fold_dir = os.path.join(dir, str(fold_num + 1))
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)

    return fold_dir


def save_model(model: torch.nn.Module, model_save_dir):
    torch.save(model.state_dict(), model_save_dir)


def load_model(model_init, model_path, device):
    model_init.load_state_dict(torch.load(model_path, map_location=device))
    model_init.eval()
    return model_init


def plot_model_metric(train_loss: List, val_mae: List, save_dir: str, task):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    epoch_list = [i for i in range(1, len(train_loss) + 1)]
    interval_list = [i for i in range(1, len(val_mae) + 1)]
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(epoch_list, train_loss)
    axs1.set(xlabel="epoch", ylabel="loss")
    axs1.set_title("Train Loss", fontsize=14)
    axs2.plot(interval_list, val_mae)
    axs2.set(xlabel="epoch / interval", ylabel="Mae (eV)")
    axs2.set_title("Valid Mae", fontsize=14)

    t = time.localtime()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{task}.jpg"),
                dpi=300)


def plot_comparison_pic(label, predict, save_dir):
    plt.style.use('_mpl-gallery-nogrid')
    plt.figure(dpi=300, figsize=(12, 12))
    font_dict = dict(fontsize=14,
                     weight='light',
                     )
    x = np.linspace(-14, 2, num=100)
    y = x
    plt.hist2d(predict, label, bins=(x, y), cmap="Blues")
    plt.plot(x, y, linestyle='--', color="r")
    plt.xlabel("ML prediction", fontdict=font_dict)
    plt.ylabel("DFT calculation", fontdict=font_dict)
    plt.xticks(range(-14, 2, 2))
    plt.yticks(range(-14, 2, 2))
    plt.tight_layout()
    t = time.localtime()
    final_results = {"label": label, "predict": predict}
    pd.DataFrame(data=final_results).to_csv(os.path.join(save_dir, "test_results.csv"), index=False)
    plt.savefig(os.path.join(save_dir, f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_comparison.jpg"),
                dpi=300)
