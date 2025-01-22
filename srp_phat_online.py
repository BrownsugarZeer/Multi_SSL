import csv
import time
import torch
import numpy as np
from multi_ssl.utils import Duet, parsing_params, doa_detection
from multi_ssl.mic import MicrophoneStream, get_microphone_chunks


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


microphone_stream = MicrophoneStream(
    rate=16000,
    chunk=1600,
    channels=6,
    ignored_channels=[0, 5],
)


def main():
    params = parsing_params()
    with open(
        f"records/{time.strftime(r'%m_%d-%H_%M_%S')}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        _header = []
        for i in range(params.src):
            _header.extend([f"s{i+1}_azi", f"s{i+1}_ele"])
        writer.writerow(_header)

        try:
            for sample_rate, waveform in get_microphone_chunks(
                microphone_stream,
                min_to_cumulate=3,
                max_to_cumulate=10,
            ):
                duet = Duet(
                    waveform.transpose(),
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
                print("=" * 51)
                writer.writerow(doas.flatten().round().tolist())

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
