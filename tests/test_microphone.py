import numpy as np
import soundfile as sf
from icecream import ic

from multi_ssl.mic.audio_recorder import get_microphone_chunks
from multi_ssl.mic.microphone_stream import MicrophoneStream


waveforms = []
microphone_stream = MicrophoneStream(
    rate=16000,
    chunk=1600,
    channels=6,
    ignored_channels=[0, 5],
)

try:
    for sample_rate, waveform in get_microphone_chunks(
        microphone_stream,
        precumulate=0,
    ):
        ic(sample_rate, waveform.shape)
        waveforms.append(waveform)
finally:
    sf.write("output.wav", np.concatenate(waveforms, axis=0), 16000, "PCM_16")
