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
(venv) > python srp_phat_online.py
Find 1 available sources.
azi:   +6.1, ele:  +12.1, xyz: [0.9723699  0.10439985 0.20879988]
Find 1 available sources.
azi:   +4.1, ele:   +8.0, xyz: [0.98768836 0.06995953 0.13991925]
Find 1 available sources.
azi:   +4.1, ele:   +8.0, xyz: [0.98768836 0.06995953 0.13991925]
...
```

2. Using an audio file (offline)

```
# Automatically determine the number of sources
(venv) > python srp_phat_offline.py --wave=data/a0e45_a-45e5.wav
Find 3 available sources.
azi:  -42.2, ele:  +19.7, xyz: [ 0.697037   -0.63311356  0.33661067]
azi:   -3.6, ele:  +57.5, xyz: [ 0.5367838  -0.03401686  0.8430338 ]
azi:  -42.6, ele:   +3.4, xyz: [ 0.7347942  -0.6757342   0.05882788]

# Force to return two sources
(venv) > python srp_phat_offline.py --wave=data/a0e45_a-45e5.wav --src=2
Find 2 available sources.
azi:  -42.6, ele:   +3.4, xyz: [ 0.7347942  -0.6757342   0.05882788]
azi:   -3.6, ele:  +57.5, xyz: [ 0.5367838  -0.03401686  0.8430338 ]
```

# Visualization

To easily show what's going on, we use [plotly](https://github.com/plotly/plotly.py) to plot the DOA on a sphere which diameter is 1 meter. The center of the sphere is the microphone array we place at p(x=0, y=0, z=0), the dark blue dots are the Directions of Arrival (DOA), and the lighter dots are the projections on each plane.

```
(venv) > python srp_visualizer.py --src=1 --wav=data/a0e55.csv
```

<img src="./img/demo_srp_phat.png" width="700">

# Issue

1. The algorithm has a high computational complexity thus making the algorithm unsuitable for real time applications. For estimating one source we need at least 0.3 seconds, estimating N sources we need at least (0.3\*N) seconds,

# References

1. Dey, Ajoy Kumar, and Susmita Saha. "[Acoustic Beamforming: Design and Development of Steered Response Power With Phase Transformation (SRP-PHAT).](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A830527&dswid=-9722)" (2011).

2. Ravanelli, Mirco, et al. "[SpeechBrain: A General-Purpose Speech Toolkit.](https://speechbrain.github.io/)" arXiv preprint arXiv:2106.04624 (2021).
