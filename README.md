# 3D Multiple Sound Sources Localization (SSL)

The Steered Response Power Phase Transform (SRP-PHAT) is an important and robust algorithm to localize acoustic sound sources. However, the algorithm can only give us one location estimation. For multi-sources extension, we propose to use the Degraded Unmixing Estimation Technique (DUET) to separate each source and pass it to the SRP-PHAT algorithm to achieve multi-sources tracking.

# Prepare an Environment

```
git clone https://github.com/BrownsugarZeer/Multi_SSL.git
cd Multi_SSL
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

> [Pyaudio](http://people.csail.mit.edu/hubert/pyaudio/) requires some tricks to install on Windows. If the installation fails, finding [unofficial wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) may be a available solution.

# Hardware

The [board](https://github.com/respeaker/usb_4_mic_array) is a far-field microphone array device capable of detecting voices up to 5m away even with the presence of background noise.\
<img src="./img/respeaker.png" width="500">

# Running an Experiment

1. Using a microphone stream (online)

```
(venv) > python srp_phat_online.py  -s=1
Find 1 available sources.
azi:  184.4, ele:   46.4
===================================================
Find 1 available sources.
azi:  184.4, ele:   46.4
===================================================
Find 1 available sources.
azi:  276.1, ele:   39.2
===================================================
...
```

2. Using an audio file (offline)

```
# Automatically determine the number of sources
(venv) > python srp_phat_offline.py -s=1 -c=4 -i=None --wave=data/a0e20/50cm/a0e19_3_1b6ede00.wav
Find 1 available sources.
azi:    0.3, ele:   22.7

(venv) > python srp_phat_offline.py -s=2 -c=4 -i=None --wave=data/a0e20_a45e35/150cm/a0e19_a44e34_3_1c91d780.wav
Find 2 available sources.
azi:   50.8, ele:   43.2
azi:    2.7, ele:   26.2
```

# Visualization

To easily show what's going on, we use [plotly](https://github.com/plotly/plotly.py) to plot the DOA on a sphere which diameter is 1 meter. The center of the sphere is the microphone array we place at p(x=0, y=0, z=0), the dark blue dots are the Directions of Arrival (DOA), and the lighter dots are the projections on each plane.

```
(venv) > python srp_visualizer.py -s=1 --wav=data/a0e20/50cm.csv
```

50cm\
<img src="./img/50cm.png" width="700">\
150cm\
<img src="./img/150cm.png" width="700">\
250cm\
<img src="./img/250cm.png" width="700">

# Issue

1. The algorithm has a high computational complexity thus making the algorithm unsuitable for real time applications. For estimating one source we need at least 0.3 seconds, estimating N sources we need at least (0.3\*N) seconds,

# References

1. S. Rickard, "[The DUET blind source separation algorithm.](https://www.researchgate.net/publication/227143748_The_DUET_blind_source_separation_algorithm)" Blind Speech Separation, pp. 217-241, 2007.

2. Dey, Ajoy Kumar, and Susmita Saha. "[Acoustic Beamforming: Design and Development of Steered Response Power With Phase Transformation (SRP-PHAT).](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A830527&dswid=-9722)" (2011).

3. Ravanelli, Mirco, et al. "[SpeechBrain: A General-Purpose Speech Toolkit.](https://speechbrain.github.io/)" arXiv preprint arXiv:2106.04624 (2021).
