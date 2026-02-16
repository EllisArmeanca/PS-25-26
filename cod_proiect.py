
# Monophonic piano note analysis: waveform, spectrogram, F0 detection, onsets, notes + durations

#importuri:
import os
from scipy.io.wavfile import read
from scipy.signal import spectrogram, medfilt
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import librosa
import librosa.display


# 1. CONFIG

fn = "doremiManual" # numele fisierului audio
audio_path = f"../samples/{fn}.wav"

# un folder pentru fiecare fisier unde sa fie salvate graficile.
fig_base_dir = f"./figures/{fn}/"
os.makedirs(fig_base_dir, exist_ok=True)

# marimea ferestrei.
# 2048 samples at 44100 Hz ~ 46 ms. bun pt comporomisul timp-frecventa
frame_size = 2048

# Pasul ferestrei
# 512 samples hop => 75% overlap (2048 - 512 = 1536 overlap).
hop_size = 512

#----------
# 2. LOAD + PREPROCESARE AUDIO
#---------
fs, x = read(audio_path)
print("Sampling rate:", fs)

# conversia stereo - mono
if x.ndim == 2:
    x = x[:, 0]

x = x.astype(np.float32) # conversia la numerele reale

x = x / np.max(np.abs(x)) # normalizarea

print("Number of samples:", x.size)
print("Timpul (secunde):", x.size / fs)

t = np.arange(x.size) / float(fs) # axa timpului in secunde pentru waveform

# --------
# 3. WAVEFORM PLOTS
# --------

# Plot waveform using plain matplotlib
plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Piano waveform")
plt.tight_layout()
plt.savefig(os.path.join(fig_base_dir, "waveform_matplotlib.png"), dpi=300)
plt.show()

# Plot waveform using librosa display helper
plt.figure(figsize=(10, 4))
librosa.display.waveshow(x, sr=fs, color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform (librosa)")
plt.tight_layout()
plt.savefig(os.path.join(fig_base_dir, "waveform_librosa.png"), dpi=300)
plt.show()

# -------
# 4. SPECTROGRAM (STFT)
# ------


stft = librosa.stft(x, n_fft=4096, hop_length=256, window="hann")# window hann pentru spectral leakage.


spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max) # spectrul de magnitudine

# Display spectrogram: time on x-axis, log-frequency on y-axis.
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    spectrogram_db,
    sr=fs,
    hop_length=256,
    x_axis="time",
    y_axis="log",
    cmap="magma"
)
plt.colorbar()
plt.title("Spectrogram (librosa, log-frequency)")
plt.tight_layout()
plt.savefig(os.path.join(fig_base_dir, "spectrogram_librosa.png"), dpi=300)
plt.show()

# ---------
# 5. FRAME + FFT
# ---------

# window hann pentru spectral leakage
window = np.hanning(frame_size)

# spectrul de magnitudine pe fiecare fereastra.
fft_frames = []


num_frames = 1 + int((len(x) - frame_size) / hop_size)

for i in range(num_frames):

    start = i * hop_size
    end = start + frame_size

    # Extragem frame
    frame = x[start:end]

    # aplicam hann
    frame = frame * window


    X = np.fft.rfft(frame) # real fft

    fft_frames.append(np.abs(X))

fft_frames = np.array(fft_frames) # (num_frames, num_bins)

freqs_per_bin = np.linspace(0, fs / 2, fft_frames.shape[1])

print("Number of frames:", fft_frames.shape[0])

# ======================
# 6. PEAK DETECTION
# ======================

def detect_peaks(mag_spectrum, threshold_ratio=0.3):

    peaks = []
    mags = []

    # Threshold to ignore small values / noise
    thr = threshold_ratio * np.max(mag_spectrum)

    # ne uitam pe spectru, fara primul si ultimul element
    for k in range(1, len(mag_spectrum) - 1):
        # maxim local
        if mag_spectrum[k] > mag_spectrum[k - 1] and mag_spectrum[k] > mag_spectrum[k + 1]:
            #  peste threshold
            if mag_spectrum[k] > thr:
                peaks.append(k)
                mags.append(mag_spectrum[k])

    return peaks, mags

peak_freqs = []  # list of peak frequencies per frame (Hz)
peak_mags = []   # list of peak magnitudes per frame

for mag in fft_frames:
    bins, mags = detect_peaks(mag)
    # Convert bin indices to frequencies in Hz
    peak_freqs.append(freqs_per_bin[bins])
    peak_mags.append(mags)

print("First frame peak freqs (Hz):", peak_freqs[0][:10])

# ======================
# 7. F0 (HPS METHOD)
# ======================

def hps_f0(mag_spectrum, fs, harmonics=5):

    # spectrul de magnitudine
    spectrum = mag_spectrum.copy()
    N = len(spectrum)

    # initializam hps  cu spectrul original
    hps = spectrum.copy()

    # Multiply with downsampled
    for h in range(2, harmonics + 1):
        down = spectrum[::h]           # downsample by factor h
        hps[:len(down)] *= down

    # Index of maximum in HPS
    idx = np.argmax(hps)

    # Coconvertim varful indexului to frequency
    f0 = idx * (fs / 2) / (N - 1)
    return f0

f0_per_frame = np.array([hps_f0(mag, fs) for mag in fft_frames])

print("First 20 raw f0 estimates:", f0_per_frame[:20])

# ---------
# 8. SMOOTHING
# ------

# Apply median filter (outliers)
f0_smooth = medfilt(f0_per_frame, kernel_size=5)

plt.figure(figsize=(10, 4))
plt.plot(f0_per_frame, label="Raw f0", alpha=0.7)
plt.plot(f0_smooth, label="Smoothed f0", linewidth=2)
plt.legend()
plt.title("F0 smoothing")
plt.tight_layout()
plt.savefig(os.path.join(fig_base_dir, "f0_smoothing.png"), dpi=300)
plt.show()

# -----
# 9. ONSET DETECTION (SPECTRAL FLUX)
# ------

# Spectral flux measures how much the spectrum changes between consecutive frames.
# Onsets usually correspond to large positive changes in the spectrum.
spectral_flux = []

for i in range(1, len(fft_frames)):
    # diff intre actual si precedent
    diff = fft_frames[i] - fft_frames[i - 1]

    # pastram doar dif pozitive
    diff[diff < 0] = 0

    # Spectral flux = sum of positive changes
    spectral_flux.append(np.sum(diff))

spectral_flux = np.array(spectral_flux)

# normalizare
spectral_flux = spectral_flux / np.max(spectral_flux)

plt.figure(figsize=(12, 4))
plt.plot(spectral_flux)
plt.title("Spectral flux")
plt.tight_layout()
plt.savefig(os.path.join(fig_base_dir, "spectral_flux.png"), dpi=300)
plt.show()

#   ONSET DETECTION

# Threshold for spectral flux peaks.
# Lower value (=0.2) is more sensitive, detects softer onsets.
threshold = 0.2

onsets = []

# We detect local maxima in the spectral flux that are above threshold.
for i in range(1, len(spectral_flux) - 1):
    if spectral_flux[i] > threshold:
        if spectral_flux[i] > spectral_flux[i - 1] and spectral_flux[i] > spectral_flux[i + 1]:
            onsets.append(i)

# For safety, we always include the first frame as an onset
# so that the first note is not missed.
if 0 not in onsets:
    onsets.insert(0, 0)

print("Onset frame indices:", onsets)

# ======================
# 10. NOTE SEGMENTATION
# ======================

# fiecare nota intre doua onset consecutive

notes = []        # list of (f0_note, start_time, end_time, duration)
hop_time = hop_size / fs  # seconds per frame

onsets_with_end = onsets + [len(f0_smooth)]

for i in range(len(onsets)):
    start_f = onsets[i]
    end_f = onsets_with_end[i + 1]

    if end_f <= start_f:
        continue

    # Extract F0 values
    f0_segment = f0_smooth[start_f:end_f]

    # Remove zero or very low F0 values
    f0_segment = f0_segment[f0_segment > 10]

    if len(f0_segment) == 0:
        continue


    f0_note = np.median(f0_segment)

    # frame -> time conversion
    start_time = start_f * hop_time
    end_time = end_f * hop_time
    duration = end_time - start_time

    notes.append((f0_note, start_time, end_time, duration))

print("\n=== Notes (F0 + time) ===")
for idx, (f0, st, en, dur) in enumerate(notes):
    print(f"Note {idx+1}: f0={f0:.2f} Hz | start={st:.2f}s | end={en:.2f}s | dur={dur:.2f}s")

# --------
# 11. MAP F0 -> MIDI NOTE
#-----

note_names = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

final_notes = []   # list of (note_name, f0, start, end, duration)

for (f0, st, en, dur) in notes:
    # Convert frequency (Hz) to MIDI note number.
    # Formula: midi = 69 + 12 * log2(f0 / 440)
    midi = 69 + 12 * np.log2(f0 / 440.0)
    midi = int(round(midi))

    # Get note name and octave from MIDI number
    name = note_names[midi % 12]
    octave = midi // 12 - 1
    note_full = f"{name}{octave}"

    final_notes.append((note_full, f0, st, en, dur))

print("\n=== DETECTED NOTES ===")
for idx, (note_txt, f0, st, en, dur) in enumerate(final_notes):
    print(f"{idx+1}. {note_txt} | f0={f0:.2f} Hz | start={st:.2f}s | end={en:.2f}s | dur={dur:.2f}s")
