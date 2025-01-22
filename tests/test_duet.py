import numpy as np
from icecream import ic
from scipy.io.wavfile import read, write
from multi_ssl.utils.bss import Duet

if __name__ == "__main__":

    fs, x = read("mixtures_trash_kaf_16000.wav")
    x = np.transpose(x)[1:5, :]
    ic(x.shape)
    duet = Duet(
        x,
        n_sources=2,
        sample_rate=fs,
        delay_max=2.0,
    )
    estimates = duet()
    ic(estimates.shape)
    for i in range(duet.n_sources):
        write(f"duet_s{i}.wav", duet.fs, estimates[i, :] + 0.05 * duet.x1)
    duet.plot_atn_delay_hist()
