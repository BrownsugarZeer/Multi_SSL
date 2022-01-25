import torch
import numpy as np
from math import atan2

from speechbrain.processing.features import STFT
from speechbrain.processing.multi_mic import Covariance, SrpPhat


def inti_mics():
    mics = torch.zeros((4, 3), dtype=torch.float)
    mics[0, :] = torch.FloatTensor([-0.02285, -0.02285, +0.005])
    mics[1, :] = torch.FloatTensor([+0.02285, -0.02285, +0.005])
    mics[2, :] = torch.FloatTensor([+0.02285, +0.02285, +0.005])
    mics[3, :] = torch.FloatTensor([-0.02285, +0.02285, +0.005])
    return mics


def doa_detection(waveform):
    """
    Using the SRP-PHAT to determine the direction of angles.

    Augments
    --------
    waveform : torch.Tensor
        the input shape is [batch, time_step, channel], and
        the number of channels should be at least 4.
    """
    mics = inti_mics()
    stft = STFT(sample_rate=16000)
    cov = Covariance()
    srpphat = SrpPhat(mics=mics)

    Xs = stft(waveform)
    XXs = cov(Xs)
    doas = srpphat(XXs)
    xyz = doas[0, 0, :].numpy()
    xyz[2] = np.abs(xyz[2])
    r = np.sqrt(xyz[0]**2 + xyz[1]**2)
    azi = atan2(xyz[1], xyz[0]) * 57.295779
    ele = atan2(np.abs(xyz[2]), r) * 57.295779
    print(f"azi: {azi:>+6.1f}, ele: {ele:>+6.1f}, xyz: {xyz}")
    return f"{azi:+.1f}", f"{ele:+.1f}"
