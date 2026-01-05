from scipy import misc, ndimage, datasets
import numpy as np
import matplotlib
from scipy.fft import dctn, idctn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# imaginea cu raton
X = datasets.ascent().astype(float)

plt.imshow(X, cmap=plt.cm.gray)
plt.title("Original")
plt.show()

#  UTILIZATORUL setează pragul MSE
prag_MSE = 200

#  Transformare DCT
Y = dctn(X)

#  Compresie prin taierea frecventelor
H, W = Y.shape
Y_flat = Y.flatten()

# sortam coeficientii dupa valoare ABS (energie)
ordine = np.argsort(np.abs(Y_flat))  # de la mic la mare

# copie pentru a face zero treptat
Y_comp = Y_flat.copy()

# cautam primul punct unde MSE < prag
for k in range(len(Y_flat)):
    # punem 0 coeficientul cel mai mic ramas
    Y_comp[ordine[k]] = 0

    # reconstructie
    Y_2D = Y_comp.reshape(H, W)
    X_rec = idctn(Y_2D)

    # calcul MSE
    mse = np.mean((X - X_rec)**2)

    print("k =", k, " MSE =", mse)

    if mse < prag_MSE:
        print(">>> Am atins pragul MSE:", mse)
        break

# ---------- Afisare rezultat ----------
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.imshow(X, cmap='gray')
plt.title("Original")

plt.subplot(122)
plt.imshow(X_rec, cmap='gray')
plt.title("Compresie cu prag MSE")

# SAVE FIGURE
plt.savefig("ex3ss1.png", dpi=300, bbox_inches="tight")

plt.show()
