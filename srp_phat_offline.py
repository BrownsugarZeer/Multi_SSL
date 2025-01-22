import torch
import numpy as np
from pathlib import Path
from scipy.io.wavfile import read

from multi_ssl.utils import Duet, parsing_params, doa_detection


def init_respeaker_mic_array(rotation=0):
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
    mics[0, :] = torch.Tensor([+0.02285, +0.02285, +0.005])
    mics[1, :] = torch.Tensor([-0.02285, +0.02285, +0.005])
    mics[2, :] = torch.Tensor([-0.02285, -0.02285, +0.005])
    mics[3, :] = torch.Tensor([+0.02285, -0.02285, +0.005])

    if rotation:
        sin = np.sin(rotation * np.pi / 180.0)
        cos = np.cos(rotation * np.pi / 180.0)
        r_mat = torch.Tensor([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])
        mics = torch.matmul(mics, r_mat)
    return mics


def main():
    params = parsing_params()

    if Path(params.wave).is_file():
        sample_rate, x = read(params.wave)
    else:
        raise FileExistsError("the file path is not correct or pass via --wave")

    params.ignored_channels = [
        int(i) for i in params.ignored_channels if str(i).isnumeric()
    ]
    mask = np.ones(params.channels, bool)
    if not params.ignored_channels == []:
        mask[params.ignored_channels] = False
    x = np.transpose(x)[mask, :]

    duet = Duet(
        x,
        n_sources=params.src,
        sample_rate=sample_rate,
        delay_max=2.0,
        n_delay_bins=50,
        output_all_channels=True,
    )
    estimates = duet()
    estimates = estimates.astype(np.float32)
    print(f"Find {len(estimates)} available sources.")

    doas = doa_detection(
        init_respeaker_mic_array(),
        torch.from_numpy(estimates),
        sample_rate=sample_rate,
    )
    doas[doas[:, 0] < 0] += torch.FloatTensor([[360, 0]])
    for doa in doas:
        print(f"azi: {doa[0]: 6.1f}, ele: {doa[1]: 6.1f}")


if __name__ == "__main__":
    main()
