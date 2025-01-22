"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

Modified to the latest version since some of the functions were deprecated.

Original code repository: 
    https://github.com/pytorch/audio/blob/master/examples/interactive_asr/vad.py#L38
"""

import numpy as np
import torch
from typing import Generator, Tuple
from collections import deque

from multi_ssl.mic.microphone_stream import MicrophoneStream

torch.set_num_threads(1)


def get_microphone_chunks(
    microphone_stream: MicrophoneStream,
    *,
    min_to_cumulate=5,
    max_to_cumulate=20,
    precumulate=5,
    speech_threshold=0.8,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Real-time VAD(voice-activity-detection) chunk catcher.

    Augments
    --------
    microphone_stream : MicrophoneStream
        the microphone stream object.
    min_to_cumulate : int
        the minimum length of waveform, default is 5 chunks.
    max_to_cumulate : int
        the maximun length of waveform, default is 20 chunks.
    precumulate : int
        the length of waveform to keep before the speech, default is 5 chunks.
    speech_threshold : float
        the threshold to decide if the chunk contains speech, default is 0.8.

    Returns
    -------
    (waveform, sample_rate) : Tuple[ndarray, int]

    Example
    -------
    >>> import soundfile as sf
    >>> import numpy as np
    >>> from multi_ssl.mic.microphone_stream import MicrophoneStream
    >>> from multi_ssl.mic.audio_recorder import get_microphone_chunks
    >>> waveforms = []
    >>> microphone_stream = MicrophoneStream(
    >>>     rate=16000,
    >>>     chunk=1600,
    >>>     channels=6,
    >>>     ignored_channels=[0, 5],
    >>> )
    >>> try:
    >>>     for sample_rate, waveform in get_microphone_chunks(
    >>>         microphone_stream,
    >>>         precumulate=0,
    >>>     ):
    >>>         print(sample_rate, waveform.shape)  # 16000, (1600, 4)
    >>>         waveforms.append(waveform)
    >>> finally:
    >>>     sf.write("output.wav", np.concatenate(waveforms, axis=0), 16000, "PCM_16")
    """

    cumulated = []
    precumulated = deque(maxlen=precumulate)

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        trust_repo=True,
    )

    with microphone_stream as stream:
        for chunk in stream:
            # NOTE: silero-vad configs
            # chunk_size = 512 if SAMPLING_RATE == 16000 else 256
            confidence = model(
                torch.from_numpy(np.copy(chunk[:512, 0])),
                stream.rate,
            ).item()
            is_speech = confidence >= speech_threshold

            if is_speech or cumulated:
                cumulated.append(chunk)
            else:
                precumulated.append(chunk)

            if (
                not is_speech
                and len(cumulated) >= min_to_cumulate
                or len(cumulated) >= max_to_cumulate
            ):
                waveform = np.concatenate((list(precumulated) + cumulated), axis=0)
                yield (stream.rate, waveform)
                cumulated = []
                precumulated = deque(maxlen=precumulate)
