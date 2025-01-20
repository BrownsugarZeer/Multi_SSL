"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

Modified to the latest version since some of the functions were deprecated.

Original code repository:
    https://github.com/pytorch/audio/blob/master/examples/interactive_asr/vad.py#L38
"""

import numpy as np
import pyaudio
import queue
import wave
from scipy import signal
from typing import List, Optional


class MicrophoneStream(object):
    """
    Opens a recording stream as a generator yielding the audio chunks.
    It supports a multiple channels and chose the channels which want
    to leave or not. For example: we have a six channels microphone
    array, the channels would be [0, 1, 2, 3, 4, 5]. if we want to
    discard channel0 and channel5, we can pass `ignored_channels=[0, 5]`
    to let the return chunk only contains [1, 2, 3, 4].

    Augments
    --------
    rate : int
        sampling rate.
    chunk : int
        Specifies the number of frames per buffer., default is mono.
    n_channels : int
        The number of channels.
    """

    def __init__(
        self,
        rate,
        chunk,
        n_channels=1,
        ignored_channels: Optional[List] = None,
        device=None,
    ):
        """
        The ratio of [chunk / rate] is the amount of time between
        audio samples - for example, with these defaults, an audio
        fragment will be processed every tenth of a second.
        """
        self._audio_interface = pyaudio.PyAudio()
        self._rate = rate
        self._chunk = chunk
        self._device = device
        self._channels = n_channels
        self._ignored_channels = ignored_channels
        self._buff = queue.Queue()  # create a thread-safe buffer of audio data.
        self.closed = True
        self._format_type = pyaudio.paFloat32
        self._audio_stream = None

        # extract the audio
        # self._recored_wav = self.prepare_file(file_name='audiotest.wav', mode='wb')

    def __enter__(self):
        # Specifies a callback function for non-blocking (callback) operation
        # Run the audio stream asynchronously to fill the buffer object. This is
        # necessary so that the input device's buffer doesn't overflow while the
        # calling thread makes network requests, etc.

        self._audio_stream = self._audio_interface.open(
            format=self._format_type,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=self._device,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
        # self._recored_wav.close()

    def prepare_file(self, file_name, mode="wb"):
        """Create a wave file to record the audio stream"""
        wavefile = wave.open(file_name, mode)
        wavefile.setnchannels(self._channels)
        wavefile.setsampwidth(self._audio_interface.get_sample_size(self._format_type))
        wavefile.setframerate(self._rate)
        return wavefile

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(np.frombuffer(in_data, dtype=np.float32))
        # self._recored_wav.writeframes(in_data)
        return None, pyaudio.paContinue

    def generator(self, resample_rate: Optional[int] = None):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                except queue.Empty:
                    break

            # reshape from (channels * chunk, ) to (channels, chunk) and ignore
            # the channels if specified. i.g. (9600, ) -> (6, 1600), if ignored
            # ch0 and ch5, the final results will be (4, 1600)
            chunk = np.reshape(chunk, (self._channels, -1), order="F")
            if self._ignored_channels is not None:
                mask = np.ones(self._channels, bool)
                mask[self._ignored_channels] = False
                chunk = chunk[mask]

            if resample_rate is None:
                yield chunk
            else:
                yield signal.resample(chunk, resample_rate, axis=1)
