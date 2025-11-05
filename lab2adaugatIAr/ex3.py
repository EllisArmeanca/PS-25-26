import matplotlib
import scipy
import sounddevice as sd

matplotlib.use('TkAgg')

import numpy as np

# a
fs = 44100             # frecventa de esantionare
N = 1600               # numar de esantioane
t = np.arange(N) / fs  # vector de timp discret
x = np.sin(2 * np.pi * 400 * t)  # semnal sinusoidal
sd.play(x,fs)
sd.wait()

#b
fs = 44100             # frecventa de esantionare
N = 3 * fs # 3 secunde
t = np.arange(N) / fs
x = np.sin(2 * np.pi * 800 * t)
sd.play(x,fs)
sd.wait()

#c
fs = 44100
#sawtooth - x(t) = 2(ft - floor(ft)) - 1
N = int(1 * fs)
t = np.arange(N) / fs
x = 2 * (240 * t - np.floor(240 * t)) - 1
sd.play(x,fs)
sd.wait()

#d
#square - oscileaza intre doua valori constante (1 si -1)
# x(T) = sign(sin(2πft))
fs = 44100
N = int(1 * fs)
t = np.arange(N) / fs
x = np.sign(np.sin(2 * np.pi * 300 * t))

sd.play(x,fs)
sd.wait()

rate = int(10e5)
scipy.io.wavfile.write('nune.wav', rate, x)

rate, y = scipy.io.wavfile.read('nune.wav')

