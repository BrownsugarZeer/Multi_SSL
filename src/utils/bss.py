import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy as sp
from pathlib import Path
from scipy.signal import convolve2d

import cProfile
import pstats
from time import strftime
from .find_peaks import find_peak_indices
# from skimage.feature import peak_local_max
np.set_printoptions(threshold=sys.maxsize)

FILEDIR = Path(__file__).resolve().parent
EPSILON = 1e-16


def performance_test(func):
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            print(f"Now use function `{func.__name__}`")
            result = func(*args, **kwargs)
        with open(
            f"perf_{strftime(r'%m_%d-%H_%M_%S')}_{func.__name__}.txt", 'w', encoding="utf-8"
        ) as stream:
            stats = pstats.Stats(pr, stream=stream)
            stats.strip_dirs().sort_stats('tottime').print_stats()
        return result
    return wrapper


def tfsynthesis(n_sources, timefreqmat, swin, hop_length, n_fft):
    """
    time-frequency synthesis\n
    TIMEFREQMAT is the complex matrix time-freq representation\n
    SWIN is the synthesis window\n
    TIMESTEP is the # of samples between adjacent time windows.\n
    NUMFREQ is the # of frequency components per time point.\n
    X contains the reconstructed signal.\n
    """
    # MATLAB and Fortran use column-major layout by default,
    # whereas C and C++ use row-major layout
    swin = np.reshape(swin, -1, 'F')

    win_length = swin.size
    _, n_fft, numtime = timefreqmat.shape

    ind = np.fmod(np.arange(win_length), n_fft)
    x = np.zeros((n_sources, (numtime-1) * hop_length + win_length))

    # Using broadcasted version can speed up about 4 times.
    # Origin code:
    # for i in range(numtime):
    #     temp = n_fft * ifft(timefreqmat[:, i]).real
    #     sind = i * hop_length
    #     for j in range(win_length):
    #         x[sind+j] = x[sind+j] + temp[ind[j]] * swin[j]
    temp = n_fft * np.fft.ifft(timefreqmat, axis=1).real
    for i in range(numtime):
        x[:, i * hop_length: (i+2) * hop_length] += temp[:, ind, i] * swin
    return x


def multichl_tfsynthesis(n_sources, timefreqmat, swin, hop_length, n_fft):
    """
    time-frequency synthesis\n
    TIMEFREQMAT is the complex matrix time-freq representation\n
    SWIN is the synthesis window\n
    TIMESTEP is the # of samples between adjacent time windows.\n
    NUMFREQ is the # of frequency components per time point.\n
    X contains the reconstructed signal.\n
    """
    swin = np.reshape(swin, -1, 'F')

    win_length = swin.size
    _, n_fft, numtime, chls = timefreqmat.shape

    ind = np.fmod(np.arange(win_length), n_fft)
    x = np.zeros((n_sources, (numtime-1) * hop_length + win_length, chls))

    temp = n_fft * np.fft.ifft(timefreqmat, axis=1).real
    for i in range(numtime):
        x[:, i * hop_length: (i+2) * hop_length, :] += temp[:, ind, i, :] * swin[..., None]
    return x


def twoDsmooth(mat, ker):
    """
    Smoothening for better identification of the peaks in a graph.
    Could have used Gaussian Kernels to do the same but it seemed
    better visual effects were given when this algorithm was followed
    ( Again, based on original CASA495) MAT is the 2D matrix to be
    smoothed. KER is either\n
    (1) a scalar\n
    (2) a matrix which is used as the averaging kernel.\n
    """
    try:
        len(ker)
        kmat = ker

    except:
        kmat = np.ones((ker, ker)) / ker**2

    kr, kc = kmat.shape
    if (kr % 2 == 0):
        kmat = convolve2d(kmat, np.ones((2, 1)), 'symm', 'same')
        kr += 1

    if (kc % 2 == 0):
        kmat = convolve2d(kmat, np.ones((1, 2)), 'symm', 'same')
        kc += 1

    rota = np.rot90(kmat, 2)
    mat = convolve2d(mat, rota, 'same', 'symm')
    return mat


class Duet(object):
    """computes the Degenerate Unmixxing Estimation Technique (DUET).

    This class computes the the Degenerate Unmixxing Estimation Technique
    of an audio signal. It supports a microphone pair inputs. (more contents)

    Arguments
    ---------
    x : ndarray
        The input audio signal with at least two channels.
    n_sources : int
        (relative `numsources` in the paper) How many sources want to
        be seperated (maximun observed sources).
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    mic_pair : tuple
        Configure which channel x1 and x2 are, assuming the input is a
        multi-channel audio. It raised an error if input is mono, i.g.
        shape=(1, :) is mono, shape=(2, :) is stereo. The default is
        None (equivalent to tuple(0, 1)).
    attenuation_max : float
        (relative `maxa` in the paper) Only consider attenuation yielding
        estimates in bounds.
    n_attenuation_bins : int
        (relative `abins` in the paper) The range of attenuation values
        distributed into bins, default is 35.
    delay_max : float
        (relative `maxd` in the paper) Only consider delay yielding
        estimates in bounds.
    n_delay_bins : int
        (relative `dbins` in the paper) The range of delay values
        distributed into bins, default is 50.
    p : int
        Weight the histogram with the symmetric attenuation estimator,
        default is 1.
    q : int
        Weight the histogram with the delay estimator, default is 0.
    output_all_channels : bool
        If True, using the mask on the whole channels, the output
        shape is (batch, time_step, n_channels). Return output shape
        (batch, time_step) if setting is False, default is False.

    Example
    -------
    >>> from duet import Duet
    >>> from scipy.io.wavfile import read, write
    >>> # x is stereo(2 channels)
    >>> sr, x = read("<FILEDIR>/x.wav")
    >>> duet = Duet(x, n_sources=5, sample_rate=sr)
    >>> estimates = duet()
    >>> for i in range(duet.n_sources):
    >>>     write(f"output{i}.wav", duet.sr, estimates[i, :]+0.05*duet.x1)
    """

    def __init__(
        self,
        x,
        n_sources,
        sample_rate,
        mic_pair=None,
        attenuation_max=0.7,
        n_attenuation_bins=35,
        delay_max=3.6,
        n_delay_bins=50,
        p=1,
        q=0,
        output_all_channels=False,
    ):
        self.x = x
        self.n_sources = n_sources
        self.sr = sample_rate
        self.mic_pair = mic_pair
        self.attenuation_max = attenuation_max
        self.n_attenuation_bins = n_attenuation_bins
        self.delay_max = delay_max
        self.n_delay_bins = n_delay_bins
        self.p = p
        self.q = q

        self.x1 = None
        self.x2 = None
        self.tf1 = None
        self.tf2 = None
        self.fmat = None
        self.symmetric_atn = None
        self.delay = None
        self.sym_atn_peak = None
        self.delay_peak = None
        self.atn_peak = None
        self.norm_atn_delay_hist = None
        self.tf_weight = None
        self.bestind = None
        self._nfft = 1024
        self._win_length = 1024
        self._hop_length = 512
        self._awin = np.hamming(1024)
        self._output_all_channels = output_all_channels

        if self.mic_pair is None:
            self.mic_pair = (0, 1)

    def __call__(self):
        return self.run(self.mic_pair)

    def run(self, *args):
        # Create the spectrogram of the Left and Right channels, and remove DC
        # component to avoid dividing by zero frequency in the delay estimation.
        self.tf1, self.tf2, self.fmat = self._contruct_histogram(*args)

        # For each time/frequency compare the phase and amplitude of the left and
        # right channels. This gives two new coordinates, instead of time-frequency
        # it is phase-amplitude differences.
        self.symmetric_atn, self.delay = self._compute_atn_delay(self.tf1, self.tf2, self.fmat)

        # Build a 2-d histogram (one dimension is phase, one is amplitude) where
        # the height at any phase/amplitude is the count of time-frequency bins that
        # have approximately that phase/amplitude.
        self.norm_atn_delay_hist, self.tf_weight = self._compute_weighted_hist(self.symmetric_atn, self.delay)

        # Find the location of peaks in the attenuation-delay plane
        self.sym_atn_peak, self.delay_peak = self._find_n_peaks(
            self.norm_atn_delay_hist, n_peaks=self.n_sources, min_dist=2, threshold=0.2,
        )

        # Assign each time-frequency frame to the nearest peak in phase/amplitude
        # space. This partitions the spectrogram into sources (one peak per source)
        self.atn_peak, self.bestind = self._convert_peaks(self.sym_atn_peak)

        # Compute masks for separation
        # (1) Create a binary mask (1 for each tf-point belonging to my source, 0 for others)
        # (2) Mask the spectrogram with the mask created in (1).
        # (3) Rebuild the original wave file from (2).
        if self._output_all_channels:
            return self._build_multichl_masks(self.atn_peak, self.bestind)

        return self._build_masks(self.atn_peak, self.bestind)

    def _contruct_histogram(self, mic_pair):
        """
        Construct the two-dimensional weighted histogram.

        Following the step.1 in the paper.

        Arguments
        ---------
        mic_pair : tuple
            Configure which channel x1 and x2 are.

        Returns
        -------
        tf1 : ndarray
            STFT of x1.
        tf2 : ndarray
            STFT of x2.
        fmat : ndarray
            frequency matrix.
        """
        # Dividing by maximum to normalise
        self.x1 = self.x[mic_pair[0]] / np.iinfo(np.int16).max  # origin code: np.iinfo(self.x1.dtype)
        self.x2 = self.x[mic_pair[1]] / np.iinfo(np.int16).max  # origin code: np.iinfo(self.x2.dtype)

        # time-freq domain
        _, _, tf1 = sp.signal.stft(self.x1, fs=self.sr, window=self._awin, nperseg=self._win_length, return_onesided=False)
        _, _, tf2 = sp.signal.stft(self.x2, fs=self.sr, window=self._awin, nperseg=self._win_length, return_onesided=False)

        # removing DC component
        # Since the scipy stft will scale the return value, in order to match the
        # paper result, it should be rescaled back to the origin. Scipy stft pass
        # the scaling == 'spectrum' and mode == 'stft', the values will multiply
        # np.sqrt(1.0 / win.sum()**2). Here are the source codes below.
        # (1) https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/spectral.py#L1174
        # (2) https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/signal/spectral.py#L1806
        tf1 = tf1[1:, :] * self._awin.sum()
        tf2 = tf2[1:, :] * self._awin.sum()

        # calculate pos/neg frequencies for later use in delay calc
        h1 = np.arange(1, (self._nfft / 2) + 1)
        h2 = np.arange(-(self._nfft / 2) + 1, 0)
        freq = np.concatenate((h1, h2)) * ((2 * np.pi) / self._nfft)

        h = np.ones((tf1.shape[1], freq.shape[0]))
        for i in range(h.shape[0]):
            h[i] *= freq
        fmat = h.transpose()

        return tf1, tf2, fmat

    def _compute_atn_delay(self, tf1, tf2, fmat):
        """
        Calculate the symmetric attenuation (alpha) and delay (delta) for each t-f point.

        Following the step.2 in the paper, 'alpha' relative symmetric attenuation and
        'delta' relative delay

        Arguments
        ---------
        tf1 : ndarray
            STFT of x1.
        tf2 : ndarray
            STFT of x2.
        fmat : ndarray
            frequency matrix.

        Returns
        -------
        alpha : ndarray
            the symmetric attenuation.
        delta : ndarray
            the relative delay.
        """
        R21 = (tf2 + EPSILON) / (tf1 + EPSILON)
        a = np.abs(R21)
        alpha = a - 1./a
        delta = -np.imag(np.log(R21)) / fmat

        return alpha, delta

    def _compute_weighted_hist(self, alpha, delta):
        """
        Calculate weighted histogram

        Following the step.3 in the paper.

        Arguments
        ---------
        alpha : ndarray
            the symmetric attenuation.
        delta : ndarray
            the relative delay.

        Returns
        -------
        A : ndarray
            a normalized 2D histogram of symmetric attenuation and delay.
        tf_weight : ndarray
            weights. (should add more explanation to this param)
        """
        h1 = np.abs(self.tf1) * np.abs(self.tf2) ** self.p
        h2 = np.abs(self.fmat) ** self.q
        tf_weight = h1 * h2

        # only consider time-freq points yielding estimates in bounds
        amask = (np.abs(alpha) < self.attenuation_max) & (np.abs(delta) < self.delay_max)
        alpha_vec = alpha[amask]
        delta_vec = delta[amask]
        tf_weight = tf_weight[amask]

        # determine histogram indices
        alphaind = np.around((self.n_attenuation_bins-1)*(alpha_vec+self.attenuation_max)/(2*self.attenuation_max))
        deltaind = np.around((self.n_delay_bins-1)*(delta_vec+self.delay_max)/(2*self.delay_max))

        # FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
        # A(alphaind(k),deltaind(k)) = tf_weight(k), S is abins-by-dbins
        A = sp.sparse.csr_matrix(
            (tf_weight, (alphaind, deltaind)), shape=(self.n_attenuation_bins, self.n_delay_bins)
        ).toarray()

        # smooth the histogram - local average 3-by-3 neighboring bins
        A = twoDsmooth(A, 3)

        return A, tf_weight

    def _find_n_peaks(self, norm_atn_delay_hist, n_peaks=None, min_dist=None, threshold=0.2):
        """
        Find the n largest peaks in the 2D histogram.

        Following the step.4 in the paper.

        Arguments
        ---------
        norm_atn_delay_hist : ndarray
            a normalized 2D histogram of symmetric attenuation and delay.
        n_peaks : int
            how many peaks should be detected.

        Returns
        -------
        atn_peak : ndarray
            an array contains the peaks of symmetric attenuation.
        delay_peak : ndarray
            an array contains the peaks of delay.
        """
        x = np.linspace(-self.delay_max, self.delay_max, self.n_delay_bins)
        y = np.linspace(-self.attenuation_max, self.attenuation_max, self.n_attenuation_bins)

        if n_peaks is None:
            peaks = 5
        else:
            peaks = n_peaks

        # Peaks: [a_idx, d_inx]
        peaks = find_peak_indices(norm_atn_delay_hist, n_peaks=peaks, min_dist=min_dist, threshold=threshold)
        peaks = np.asarray(peaks)

        # Alternative method:
        # peaks = peak_local_max(norm_atn_delay_hist, num_peaks=peaks)

        cand_peaks = norm_atn_delay_hist[peaks[:, 0], peaks[:, 1]]
        if n_peaks is None:
            std = np.sqrt((np.abs(cand_peaks - cand_peaks[0])**2).mean())
            cand_peaks = cand_peaks[np.abs(cand_peaks - cand_peaks[0]) < std*0.7]

        atn_peak = y[peaks[:cand_peaks.size, 0]]
        delay_peak = x[peaks[:cand_peaks.size, 1]]

        # Debug:
        # for a_idx, d_inx in peaks:
        #     print(f"(attenuation, delay) : ({y[a_idx]:>5.2f}, {x[d_inx]:5.2f})")  # debug

        return atn_peak, delay_peak

    def _convert_peaks(self, sym_atn_peak):
        """
        Determine masks for separation.

        Following the step.5 in the paper.

        Arguments
        ---------
        sym_atn_peak : ndarray
            an array contains the peaks of symmetric attenuation.

        Returns
        -------
        peaka : ndarray
            an array contains the peaks of attenuation.
        bestind : ndarray
            (should add more explanation to this param)
        """
        # convert the symmetric attenuation back to attenuation
        peaka = (sym_atn_peak + np.sqrt(np.square(sym_atn_peak) + 4)) / 2
        bestsofar = float("inf") * np.ones(self.tf1.shape)
        bestind = np.zeros(self.tf1.shape)

        for i in range(sym_atn_peak.size):
            score = (
                np.abs(peaka[i] * np.exp(-1j*self.fmat*self.delay_peak[i]) * self.tf1 - self.tf2) ** 2
            ) / (1 + peaka[i] ** 2)
            mask = score < bestsofar
            s_mask = score[mask]
            np.place(bestind, mask, i+1)
            np.place(bestsofar, mask, s_mask)

        return peaka, bestind

    def _build_masks(self, atn_peak, bestind):
        """
        Demix with ML alignment and convert to time domain.

        Following the step.6 and step.7 in the paper.

        Arguments
        ---------
        atn_peak : ndarray
            an array contains the peaks of attenuation.
        bestind : ndarray
            (should add more explanation to this param)

        Returns
        -------
        est : ndarray
            an array contains a seperated wave stream of all speakers.
            the shape is (batch, time_step)
        """
        # 'h' stands for helper, we're using helper variables to break down
        # the logic of what's going on. Apologies for the order of the 'h's
        # Broadcast(a bit faster) the n_sources estimations and return directly.
        # h1     -> (1, 129)
        # h3     -> (1,) * (1023, 129) * (1023, 129)
        # h4     -> (1023, 129) / (1,)
        # h2     -> (1023, 129) * (1023, 129)
        # h      -> (1+1023, 129)
        # new_h1 -> (n_src, 1, 129)
        # new_h3 -> (n_src, None, None) * (n_src, 1023, 129) * (None, 1023, 129)
        # new_h4 -> (None, 1023, 129) / (n_src,)
        # new_h2 -> (n_src, 1023, 129) * (n_src, 1023, 129)
        # new_h  -> (n_src, 1+1023, 129)
        #
        # Origin code:
        # est = np.zeros((self.n_sources, self.x1.shape[-1]))
        # for i in range(self.n_sources):
        #     mask = (bestind == i+1)
        #     h1 = np.zeros((1, self.tf1.shape[-1]))
        #     h3 = atn_peak[i] * np.exp(1j*self.fmat*self.delay_peak[i]) * self.tf2
        #     h4 = ((self.tf1+h3) / (1+atn_peak[i]**2))
        #     h2 = h4 * mask
        #     h = np.concatenate((h1, h2))
        #
        #     esti = tfsynthesis(h, np.sqrt(2)*self._awin/1024, self._hop_length, self._nfft)
        #
        #     # add back into the demix a little bit of the mixture
        #     # as that eliminates most of the masking artifacts
        #     est[i] = esti[0:self.x1.shape[-1]]
        #     write(f"out{i}.wav", self.sr, est[i]+0.05*self.x1)

        h3 = (atn_peak[:, None, None]
              * np.exp(1j * self.fmat[None, ...] * self.delay_peak[:, None, None])
              * self.tf2[None, ...])
        h4 = ((self.tf1[None, ...] + h3) / (1 + atn_peak[:, None, None] ** 2))

        # In order to avoid errors caused by the observed source
        # being less than the source we set.
        observed_src = h4.shape[0]
        mask = np.zeros((observed_src, bestind.shape[0], bestind.shape[1]))
        for i in range(observed_src):
            mask[i, ...] = (bestind == i+1)

        h1 = np.zeros((observed_src, 1, self.tf1.shape[-1]))
        h2 = h4 * mask
        h = np.concatenate((h1, h2), axis=1)

        est = tfsynthesis(observed_src, h, np.sqrt(2)*self._awin/1024, self._hop_length, self._nfft)

        return est[:, 0:self.x1.shape[-1]]

    def _build_multichl_masks(self, atn_peak, bestind):
        """
        Demix with ML alignment and convert to time domain.

        Following the step.6 and step.7 in the paper.

        Arguments
        ---------
        bestind : ndarray
            (should add more explanation to this param)

        Returns
        -------
        est : ndarray
            an array contains a seperated wave stream of all speakers.
            the shape is (batch, time_step, n_channels)
        """
        # 'h' stands for helper, we're using helper variables to break down
        # the logic of what's going on. Apologies for the order of the 'h's
        # Broadcast(a bit faster) the n_sources estimations and return directly.
        # mask -> (n_src, freq, time_step)
        # h3   -> (freq, time_step, n_channels)
        # h1   -> (n_src, 1, time_step, n_channels)  # DC component
        # h2   -> (None, freq, time_step, n_channels) * (n_src, freq, time_step, None)
        # h    -> (n_src, 1+freq, time_step, n_channels)

        # Dividing by maximum to normalise
        x1 = self.x[0] / np.iinfo(np.int16).max
        x2 = self.x[1] / np.iinfo(np.int16).max
        x3 = self.x[2] / np.iinfo(np.int16).max
        x4 = self.x[3] / np.iinfo(np.int16).max

        # time-freq domain
        _, _, tf1 = sp.signal.stft(x1, fs=self.sr, window=self._awin, nperseg=self._win_length, return_onesided=False)
        _, _, tf2 = sp.signal.stft(x2, fs=self.sr, window=self._awin, nperseg=self._win_length, return_onesided=False)
        _, _, tf3 = sp.signal.stft(x3, fs=self.sr, window=self._awin, nperseg=self._win_length, return_onesided=False)
        _, _, tf4 = sp.signal.stft(x4, fs=self.sr, window=self._awin, nperseg=self._win_length, return_onesided=False)

        # removing DC component
        tf1 = tf1[1:, :] * self._awin.sum()
        tf2 = tf2[1:, :] * self._awin.sum()
        tf3 = tf3[1:, :] * self._awin.sum()
        tf4 = tf4[1:, :] * self._awin.sum()

        h3 = np.stack([tf1, tf2, tf3, tf4], axis=-1)

        observed_src = atn_peak.shape[0]
        mask = np.zeros((observed_src, bestind.shape[0], bestind.shape[1]))
        for i in range(observed_src):
            mask[i, ...] = (bestind == i+1)

        h1 = np.zeros((observed_src, 1, self.tf1.shape[-1], h3.shape[-1]))
        h2 = h3[None, ...] * mask[..., None]
        h = np.concatenate((h1, h2), axis=1)

        est = multichl_tfsynthesis(observed_src, h, np.sqrt(2)*self._awin/1024, self._hop_length, self._nfft)

        return est[:, 0:self.x1.shape[-1], :]

    def plot_atn_delay_hist(self, save_path=None):
        if self.norm_atn_delay_hist is None:
            raise RuntimeError("It should compute a weighted histogram first.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.linspace(-self.delay_max, self.delay_max, self.n_delay_bins)
        Y = np.linspace(-self.attenuation_max, self.attenuation_max, self.n_attenuation_bins)
        X, Y = np.meshgrid(X, Y)
        ax.plot_wireframe(X, Y, self.norm_atn_delay_hist)

        if save_path is not None and Path(save_path).parent.is_dir():
            plt.savefig(f'{Path(save_path)}', dpi=300)
        plt.show()
