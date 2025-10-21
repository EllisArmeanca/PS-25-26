import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

t= np.linspace(0, 0.03,61) #creeaza un sir de valori intre start si stop cu un numar fix de puncte incluzand captele
#arrange
x_t = x(t) # vectorii cu valorile semnalelor in fiecare moment de timp din t
y_t = y(t)
z_t = z(t)

fs = 200
T = 1/fs

n = np.arange(0, int(0.03/T + 1))

# Momente discrete
t_n = n * T
x_n = x(t_n)
y_n = y(t_n)
z_n = z(t_n)


fig, axs = plt.subplots(3)
fig.suptitle('Titlu principal')

axs[0].plot(t, x_t)
axs[0].stem(t_n, x_n, basefmt=' ') #t_n pt ca folosim aceeasi axa de timp
axs[0].set_title('x(t)')
axs[0].grid(True)

axs[1].plot(t, y_t)
axs[1].stem(t_n, y_n, basefmt=' ') #t_n pt ca folosim aceeasi axa de timp
axs[1].set_title('y(t)')
axs[1].grid(True)

axs[2].plot(t, z_t)
axs[2].stem(t_n, z_n, basefmt=' ') #t_n pt ca folosim aceeasi axa de timp
axs[2].set_title('z(t)')
axs[2].grid(True)

for ax in axs.flat:
    ax.set_xlim([0, 0.03])

plt.tight_layout()
plt.savefig("smenale_continue.pdf", dpi=300)
plt.show()
#daca vrei ceva neted trebuie sa esantionezi foarte foarte repede


fig2, axs2 = plt.subplots(3)
fig2.suptitle('Semnale discrete (fs = 200 Hz)')

# x[n]
axs2[0].stem(n, x_n)
axs2[0].set_title('x[n]')
axs2[0].set_xlabel('n (indexul esantionului)')
axs2[0].set_ylabel('Amplitudine')
axs2[0].grid(True)

# y[n]
axs2[1].stem(n, y_n)
axs2[1].set_title('y[n]')
axs2[1].set_xlabel('n (indexul esantionului)')
axs2[1].set_ylabel('Amplitudine')
axs2[1].grid(True)

# z[n]
axs2[2].stem(n, z_n)
axs2[2].set_title('z[n]')
axs2[2].set_xlabel('n (indexul esantionului)')
axs2[2].set_ylabel('Amplitudine')
axs2[2].grid(True)

plt.tight_layout()
plt.savefig("semnale_discrete.pdf", dpi=300)
plt.show()
