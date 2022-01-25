"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

Modified to the latest version since some of the functions were deprecated.

Original code repository: 
    https://github.com/pytorch/audio/blob/master/examples/interactive_asr/vad.py#L38

References:
    [1] M.H. Moattar and M.M. Homayounpour, "A SIMPLE BUT EFFICIENT 
    REAL-TIME VOICE ACTIVITY DETECTION ALGORITHM", EUSIPCO 2009
    Available: https://ieeexplore.ieee.org/document/7077834
"""


import numpy as np


class VoiceActivityDetection(object):
    """
    There are three criteria to decide if a frame contains speech: energy, most
    dominant frequency, and spectral flatness. If any two of those are higher than
    a minimum plus a threshold, then the frame contains speech.  In the offline
    case, the list of frames is postprocessed to remove too short silence and
    speech sequences. In the online case here, inertia is added before switching
    from speech to silence or vice versa.

    Augments
    --------
    num_init_frames : int
        take the first n frames as silece, default is 30.
    ignore_silent_count : int
        still treat silence as speech if ran less than counter, default is 4.
    ignore_speech_count : int
        still treat speech as silence if ran less than counter, default is 1.
    energy_prim_thresh : int
        mark as a speech frame if get over the threshold, default is 20.
    frequency_prim_thresh : int
        mark as a speech frame if get over the threshold, default is 10.
    spectral_flatness_prim_thresh : int
        mark as a speech frame if get over the threshold, default is 3.

    Returns
    -------
    speech_mark : bool | silence_mark : bool
    """

    def __init__(
        self,
        num_init_frames=30,
        ignore_silent_count=4,
        ignore_speech_count=1,
        energy_prim_thresh=20,
        frequency_prim_thresh=10,
        spectral_flatness_prim_thresh=3,
    ):

        self.num_init_frames = num_init_frames
        self.ignore_silent_count = ignore_silent_count
        self.ignore_speech_count = ignore_speech_count

        self.energy_prim_thresh = energy_prim_thresh
        self.frequency_prim_thresh = frequency_prim_thresh
        self.spectral_flatness_prim_thresh = spectral_flatness_prim_thresh

        self.speech_mark = True
        self.silence_mark = False

        self.silent_count = 0
        self.speech_count = 0
        self.n = 0
        self.min_energy = None
        self.min_frequency = None
        self.min_spectral_flatness = None

    def _compute_spectral_flatness(self, frame, epsilon=0.01):
        # epsilon protects against log(0)
        geometric_mean = np.exp(np.log(frame + epsilon).mean(-1)) - epsilon
        arithmetic_mean = frame.mean(-1)
        return -10 * np.log10(epsilon + geometric_mean / arithmetic_mean)

    def iter(self, frame):
        frame_fft = np.fft.rfft(frame)
        amplitudes = np.abs(frame_fft)
        energy = (frame**2).sum(-1)
        frequency = amplitudes.argmax()

        spectral_flatness = self._compute_spectral_flatness(amplitudes)

        if self.n == 0:
            self.min_energy = energy
            self.min_frequency = frequency
            self.min_spectral_flatness = spectral_flatness
        elif self.n < self.num_init_frames:
            self.min_energy = min(energy, self.min_energy)
            self.min_frequency = min(frequency, self.min_frequency)
            self.min_spectral_flatness = min(
                spectral_flatness, self.min_spectral_flatness
            )

        self.n += 1

        # Add 1. to protect against log(0)
        thresh_energy = self.energy_prim_thresh * np.log(1.0 + self.min_energy)
        thresh_frequency = self.frequency_prim_thresh
        thresh_spectral_flatness = self.spectral_flatness_prim_thresh

        # Check all three conditions
        counter = 0
        if energy - self.min_energy >= thresh_energy:
            counter += 1
        if frequency - self.min_frequency >= thresh_frequency:
            counter += 1
        if spectral_flatness - self.min_spectral_flatness >= thresh_spectral_flatness:
            counter += 1

        if counter > 1:
            # Speech detected
            self.speech_count += 1
            # Inertia against switching
            if (
                self.n >= self.num_init_frames
                and self.speech_count <= self.ignore_speech_count
            ):
                # Too soon to change
                # print(f'\r(Speech__{self.speech_count:<3d}) [-] silence', end='')
                return self.silence_mark
            else:
                self.silent_count = 0
                # print(f'\r(Speech__{self.speech_count:<3d}) [O] speech', end='')
                return self.speech_mark
        else:
            # Silence detected
            self.min_energy = (
                ((self.silent_count * self.min_energy) + energy)
                / (self.silent_count + 1)
            )
            self.silent_count += 1
            # Inertia against switching
            if (
                self.n >= self.num_init_frames
                and self.silent_count <= self.ignore_silent_count
            ):
                # Too soon to change
                # print(f'\r(Silence_{self.silent_count:<3d}) [O] speech', end='')
                return self.speech_mark
            else:
                self.speech_count = 0
                # print(f'\r(Silence_{self.silent_count:<3d}) [-] silence', end='')
                return self.silence_mark
