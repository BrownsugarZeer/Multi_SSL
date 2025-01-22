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
from typing import List


class MicrophoneStream:
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
        Specifies the number of frames per buffer, default is mono channel.
    channels : int
        The number of channels.
    ignored_channels : Optional[List]
        The list of channels to ignore.
    device : int
        The device index to use, default is None.
    """

    def __init__(
        self,
        rate: int,
        chunk: int,
        channels: int = 1,
        ignored_channels: List[str] | None = None,
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
        self._channels = channels
        self._ignored_channels = ignored_channels
        self._buff = queue.Queue()  # create a thread-safe buffer of audio data.
        self._closed = True
        self._format_type = pyaudio.paFloat32
        self._audio_stream = None

    @property
    def rate(self):
        return self._rate

    @property
    def chunk(self):
        return self._chunk

    @property
    def channels(self):
        return self._channels

    @property
    def ignored_channels(self):
        return self._ignored_channels

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
        self._closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(np.frombuffer(in_data, dtype=np.float32))
        return None, pyaudio.paContinue

    def __iter__(self):
        while not self._closed:
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

            # reshape from (channels * chunk, ) to (chunk, channels) and ignore
            # the channels if specified. i.g. (9600, ) -> (1600, 6), if ignored
            # ch0 and ch5, the final results will be (1600, 4)
            chunk = np.reshape(chunk, (-1, self._channels))
            if self._ignored_channels is not None:
                mask = np.ones(self._channels, bool)
                mask[self._ignored_channels] = False
                chunk = chunk[:, mask]

            yield chunk
