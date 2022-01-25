import csv
import time
import torch
import numpy as np

from src.utils.bss import Duet
from src.utils.cmd_parser import parsing_params
from src.utils.ssl import doa_detection
from src.mic.audio_recorder import get_microphone_chunks


def main():
    params = parsing_params()
    try:
        with open(f"records/{time.strftime(r'%m_%d-%H_%M_%S')}.csv", "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["s1_azi", "s1_ele", "s2_azi", "s2_ele", "s3_azi", "s3_ele", "s4_azi", "s4_ele"])

            for (waveform, sample_rate) in get_microphone_chunks(
                rate=16000,
                chunk=1600,
                n_channels=params.channels,
                ignored_channels=params.ignored_channels,
                min_to_cumulate=3,
                max_to_cumulate=10,
            ):
                duet = Duet(
                    waveform,
                    n_sources=params.src,
                    sample_rate=sample_rate,
                    delay_max=2.0,
                    n_delay_bins=50,
                    output_all_channels=True,
                )
                estimates = duet()
                estimates = estimates.astype(np.float32)
                print(f"Find {len(estimates)} available sources.")

                src = []
                for wav in estimates:
                    src.extend(
                        doa_detection(torch.from_numpy(wav).unsqueeze(0))
                    )

                writer.writerow(src)

    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()
