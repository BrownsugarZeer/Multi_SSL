import torch
from speechbrain.processing.features import STFT
from speechbrain.processing.multi_mic import Covariance, SrpPhat


def doa_detection(mics, waveform, sample_rate=16000):
    """
    Using the SRP-PHAT to determine the direction of angles.
    Augments
    --------
    mics : torch.Tensor
        The cartesian coordinates (xyz) in meters of each microphone.
        The tensor must have the following format (n_mics, 3).
    waveform : torch.Tensor
        the input shape is [batch, time_step, channel], and
        the number of channels should be at least 4.
    sample_rate : int
        the sample rate of the input waveform.
    """
    stft = STFT(sample_rate=sample_rate)
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
