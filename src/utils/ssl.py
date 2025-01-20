import numpy as np
import torch
from speechbrain.processing.features import STFT
from speechbrain.processing.multi_mic import Covariance, SrpPhat


def inti_mics(rotation=0):
    """
    Initialize the microphone array coordinates in Euclidean coordinates.

    Augments
    --------
    rotation : int
        Use a rotation matrix to rotate the microphone array
        coordinates counterclockwise.

    Returns
    -------
    mics : Tensor
        Return the coordinates of (x, y, z).
    """

    mics = torch.zeros((4, 3))
    mics[0, :] = torch.Tensor([-0.02285, -0.02285, +0.005])
    mics[1, :] = torch.Tensor([+0.02285, -0.02285, +0.005])
    mics[2, :] = torch.Tensor([+0.02285, +0.02285, +0.005])
    mics[3, :] = torch.Tensor([-0.02285, +0.02285, +0.005])

    if rotation:
        sin = np.sin(rotation * np.pi / 180.0)
        cos = np.cos(rotation * np.pi / 180.0)
        r_mat = torch.Tensor([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])
        mics = torch.matmul(mics, r_mat)
    return mics


def doa_detection(waveform, mics=None):
    """
    Using the SRP-PHAT to determine the direction of angles.
    Augments
    --------
    waveform : torch.Tensor
        the input shape is [batch, time_step, channel], and
        the number of channels should be at least 4.
    """
    if mics is None:
        mics = inti_mics()
    stft = STFT(sample_rate=16000)
    cov = Covariance()
    srpphat = SrpPhat(mics=mics)

    Xs = stft(waveform)
    XXs = cov(Xs)
    doas = srpphat(XXs)
    doas = doas[:, 0, :]
    doas[:, 2] = doas[:, 2].abs()
    r = (doas[:, 0] ** 2 + doas[:, 1] ** 2).sqrt()
    azi = torch.atan2(doas[:, 1], doas[:, 0]).rad2deg()
    ele = torch.atan2(doas[:, 2], r).rad2deg()

    return torch.column_stack((azi, ele))
