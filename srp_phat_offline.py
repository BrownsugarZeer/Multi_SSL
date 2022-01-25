import torch
import numpy as np
from pathlib import Path
from scipy.io.wavfile import read

from src.utils.bss import Duet
from src.utils.cmd_parser import parsing_params
from src.utils.ssl import doa_detection

# Tested wave files
# 1. data/a0e45_a-45e5.wav
# 2. data/a0e45_a-135e5.wav
# 3. data/a0e45_a-45e15_a90e0.wav


def main():
    params = parsing_params()

    if Path(params.wave).is_file():
        sr, x = read(params.wave)
    else:
        raise FileExistsError("the file path is not correct or pass via --wave")

    if params.ignored_channels is not None:
        mask = np.ones(params.channels, bool)
        mask[params.ignored_channels] = False
        x = np.transpose(x)[mask, :]

    duet = Duet(
        x,
        n_sources=params.src,
        sample_rate=sr,
        delay_max=2.0,
        n_delay_bins=50,
        output_all_channels=True,
    )
    estimates = duet()
    estimates = estimates.astype(np.float32)
    print(f"Find {len(estimates)} available sources.")
    for wav in estimates:
        doa_detection(torch.from_numpy(wav).unsqueeze(0))


if __name__ == '__main__':
    main()
