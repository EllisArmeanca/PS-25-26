
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import math
import time
import wavfile

import numpy as np
wav_path = "vocalevocal.wav"   # schimba cu numele fisierului tau
fs, x = wavfile.read(wav_path)


N = len(x)

# 1% din N (cu o limita minima ca sa nu fie prea mica)
win_len = max(256, int(0.01 * N))
hop = win_len // 2                 # 50% overlap
win = np.hanning(win_len)          # fereastra Hann (reduce scurgerile)

# numar de cadre
num_frames = 1 + (N - win_len) // hop if N >= win_len else 1

# ====== (c) FFT pe fiecare grup (cadru) ======
# folosim rfft -> doar frecventele pozitive
num_bins = win_len // 2 + 1
S = np.zeros((num_bins, num_frames), dtype=np.float32)  # (frecventa, timp)

for i in range(num_frames):
    start = i * hop
    frame = x[start:start + win_len]
    if len(frame) < win_len:
        # completam ultima fereastra daca e nevoie
        fpad = np.zeros(win_len, dtype=frame.dtype)
        fpad[:len(frame)] = frame
        frame = fpad
    frame_w = frame * win
    X = np.fft.rfft(frame_w)
    S[:, i] = np.abs(X)

# ====== (d) matricea are pe coloane FFT-urile (valoare absoluta) ======
# Deja S are forma (num_bins, num_frames) si contine modulele

# ====== (e) afisare "spectrograma" (similar cu figura din curs) ======
# axa timp si frecventa
t = np.arange(num_frames) * hop / fs
f = np.linspace(0, fs/2, num_bins)

plt.figure(figsize=(10, 6))
# folosim log1p pentru dynamic range mai ok (poate folosi si 20*log10)
plt.imshow(np.log1p(S), origin="lower", aspect="auto",
           extent=[t[0], t[-1] if len(t) > 1 else (win_len/fs),
                   f[0], f[-1]])
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrograma (STFT manual, 1% window, 50% overlap)")
plt.colorbar(label="log(1+|X|)")
plt.tight_layout()
plt.savefig("spectrogram_ex6.png", dpi=200, bbox_inches="tight")
plt.show()
