from scipy import misc, ndimage, datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')        # setam backend inainte de pyplot
import matplotlib.pyplot as plt

# functie care afiseaza imaginea si spectrul ei
def plot_x_and_fft(X, title=""):
    # afisam imaginea
    plt.imshow(X, cmap=plt.cm.gray)
    plt.title("IMAGINE PRODUSA " + title)
    plt.show()

    # calculam spectrul in domeniul frecventa
    Y = np.fft.fft2(X)
    freq_db = 20*np.log10(abs(Y))

    # afisam spectrul
    plt.imshow(freq_db)
    plt.title("SPECTRU " + title)
    plt.colorbar()
    plt.show()

# X = datasets.face(gray=True) # semnalul in domeniul timp
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()

# Y = np.fft.fft2(X) # semnalul in domeniul frecventa

N= 64 #definim dimensiunea imaginii
n1 = np.arange(N)
n2 = np.arange(N)

# X1 = np.zeros((N, N))

# for i in range(N):
#     for j in range(N):
#         X1[i, j] = np.sin(2*np.pi * j/N + 3*np.pi * i/N)

# grid 2D: pentru fiecare (n2, n1) avem cate o valoare de x[n1, n2]
N1, N2 = np.meshgrid(n1, n2)
#imi creez doua matrici, una cu liniile 0.. 63, alta cu coloanele 0..64

# x1(n1,n2) = sin(2π * n1 / N + 3π * n2 / N)
X1 = np.sin(2*np.pi * N1/N + 3* np.pi * N2/N ) #impart la N pt ca folosesc frecvente normalizate
plot_x_and_fft(X1, "(x1)")

# x2(n1,n2) = sin(4π * n1 / N) + cos(6π * n2 / N)
X2 = np.sin(4*np.pi * N1/N) + np.cos(6*np.pi * N2/N)
plot_x_and_fft(X2, "(x2)")

Y = np.zeros((N,N))
Y[0,5] = 1
Y[0, N-5] = 1
X = np.fft.ifft2(Y) # din frecventa in timp
plot_x_and_fft(np.real(X), "(x3)")

Y = np.zeros((N,N))
Y[5,0] = 1
Y[N-5,0] = 1
X = np.fft.ifft2(Y)
plot_x_and_fft(np.real(X), "(x4)")

Y = np.zeros((N,N))
Y[5,5] = 1
Y[N-5,N-5] = 1
X = np.fft.ifft2(Y)
plot_x_and_fft(np.real(X), "(x5)")


