# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch

import matplotlib as mpl
mpl.use('TkAgg')


def save_plot(tensor, savepath):
    tensor = tensor.squeeze().cpu()
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def save_audio(file_path, sampling_rate, audio):
    audio = np.clip(audio.cpu().squeeze().numpy(), -0.999, 0.999)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))


def minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = 2 * (tensor - vmin) / (vmax - vmin) - 1
    return tensor


def reverse_minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1) / 2
    tensor = tensor * (vmax - vmin) + vmin
    return tensor