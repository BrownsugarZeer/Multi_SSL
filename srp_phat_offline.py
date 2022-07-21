import torch
import numpy as np
from pathlib import Path
from scipy.io.wavfile import read

from src.utils.bss import Duet
from src.utils.cmd_parser import parsing_params
from src.utils.ssl import doa_detection


def main():
    params = parsing_params()

    if Path(params.wave).is_file():
        fs, x = read(params.wave)
    else:
        raise FileExistsError("the file path is not correct or pass via --wave")

    params.ignored_channels = [int(i) for i in params.ignored_channels if str(i).isnumeric()]
    mask = np.ones(params.channels, bool)
    if not params.ignored_channels == []:
        mask[params.ignored_channels] = False
    x = np.transpose(x)[mask, :]

    duet = Duet(
        x,
        n_sources=params.src,
        sample_rate=fs,
        delay_max=2.0,
        n_delay_bins=50,
        output_all_channels=True,
    )
    estimates = duet()
    estimates = estimates.astype(np.float32)
    print(f"Find {len(estimates)} available sources.")
    doas = doa_detection(torch.from_numpy(estimates))
    doas[doas[:, 0] < 0] += torch.FloatTensor([[360, 0]])
    for doa in doas:
        print(f"azi: {doa[0]: 6.1f}, ele: {doa[1]: 6.1f}")


if __name__ == '__main__':
    main()
