"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

Modified to the latest version since some of the functions were deprecated.

Original code repository: 
    https://github.com/pytorch/audio/blob/master/examples/interactive_asr/vad.py#L38
"""

import numpy as np
from collections import deque

from .microphone_stream import MicrophoneStream
from .simple_vad import VoiceActivityDetection


def get_microphone_chunks(
    min_to_cumulate=5,  # 0.5 seconds
    max_to_cumulate=20,  # 2.0 seconds
    precumulate=5,
    max_to_visualize=20,
    resample_rate=None,
    **kwargs,
):
    """
    Real-time VAD(voice-activity-detection) chunk catcher.

    Augments
    --------
    min_to_cumulate : int
        the minimum length of waveform, default is 0.5 seconds.
    max_to_cumulate : int
        the maximun length of waveform, default is 2.0 seconds.
    precumulate : int
    max_to_visualize : int
        the initial length of waveform, default is 2.0 seconds.
    resample_rate : int:
        if resample_rate is None, that means the resample_rate is 16000 as default.

    Returns
    -------
    (waveform, sample_rate) : Tuple[ndarray, int]

    Example
    -------
    >>> import soundfile as sf
    >>> from speech_assistance.utils.audio_recorder import get_microphone_chunks
    >>> for sample_rate, waveform in get_microphone_chunks(
    >>>     rate=16000, chunk=1600, n_channels=6, ignored_channels=[0, 5]
    >>> ):
    >>>     # print(waveform)
    >>>     # print(sample_rate)
    >>>     sf.write('output.wav', waveform, 16000, 'PCM_16')
    """

    vad = VoiceActivityDetection()
    cumulated = []
    precumulated = deque(maxlen=precumulate)

    with MicrophoneStream(**kwargs) as stream:

        audio_generator = stream.generator(resample_rate=resample_rate)
        chunk_length = stream._chunk
        waveform = np.zeros((max_to_visualize * chunk_length))

        for chunk in audio_generator:
            # vad.iter(...) should be pass with 'float32'
            is_speech = vad.iter(chunk[0])

            if is_speech or cumulated:
                cumulated.append(chunk)
            else:
                precumulated.append(chunk)

            if (not is_speech and len(cumulated) >= min_to_cumulate) or (
                len(cumulated) > max_to_cumulate
            ):
                waveform = np.concatenate((list(precumulated) + cumulated), axis=-1)
                yield (stream._rate, waveform * stream._rate)
                cumulated = []
                precumulated = deque(maxlen=precumulate)
