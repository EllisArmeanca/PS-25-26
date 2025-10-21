# === Backend compatibil cu plt.show() pe Windows/PyCharm ===
import matplotlib
matplotlib.use('TkAgg')   # forteaza backend-ul TkAgg

# === Importuri de baza ===
import numpy as np
import matplotlib.pyplot as plt

# === 1) Definim semnalele continue ca functii de timp ===
def x(t):
    # cos(520*pi*t + pi/3)
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    # cos(280*pi*t - pi/3)
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t):
    # cos(120*pi*t + pi/3)
    return np.cos(120 * np.pi * t + np.pi / 3)

# === 2) (a) Simulam axa reala de timp ===
# Intervalul cerut: [0, 0.03]
# Pas mic ~ 0.0005 s (ex: 0:0.0005:0.03) pentru curbe netede
dt = 0.0005
t = np.arange(0, 0.03 + dt, dt)  # includem capatul 0.03

# === 3) Evaluam semnalele continue pe axa t ===
x_t = x(t)  # valori pentru x(t)
y_t = y(t)  # valori pentru y(t)
z_t = z(t)  # valori pentru z(t)

# === 4) (c) Setam parametrii de esantionare ===
fs = 200          # frecventa de esantionare [Hz]
T = 1 / fs        # perioada de esantionare [s]
# Numarul de esantioane din [0, 0.03] (inclusiv marginea)
n = np.arange(0, int(0.03 / T) + 1)  # indici 0,1,2,...,floor(0.03/T)
t_n = n * T       # momentele discrete in secunde

# === 5) Valorile discrete (esantionate) ===
x_n = x(t_n)      # x[n] = x(nT)
y_n = y(t_n)      # y[n] = y(nT)
z_n = z(t_n)      # z[n] = z(nT)

# === 6) (b) Figura 1: semnalele continue in 3 subplots ===
fig1, axs1 = plt.subplots(3, figsize=(8, 6))  # 3 randuri, 1 coloana
fig1.suptitle('Semnale continue x(t), y(t), z(t)')

axs1[0].plot(t, x_t)      # grafic pentru x(t)
axs1[0].set_title('x(t)')
axs1[0].grid(True)
axs1[0].set_xlim([0, 0.03])

axs1[1].plot(t, y_t)      # grafic pentru y(t)
axs1[1].set_title('y(t)')
axs1[1].grid(True)
axs1[1].set_xlim([0, 0.03])

axs1[2].plot(t, z_t)      # grafic pentru z(t)
axs1[2].set_title('z(t)')
axs1[2].grid(True)
axs1[2].set_xlim([0, 0.03])

fig1.tight_layout()       # aranjare estetica a subplot-urilor
fig1.savefig('semnale_continue.pdf', dpi=300)  # 💾 salveaza PDF
plt.show()                # afiseaza fereastra cu figura 1

# === 7) (c) Figura 2: semnalele discrete (stem) in 3 subplots ===
fig2, axs2 = plt.subplots(3, figsize=(8, 6))
fig2.suptitle('Semnale discrete x[n], y[n], z[n] (fs = 200 Hz)')

# x[n]
axs2[0].stem(n, x_n, basefmt=' ')
axs2[0].set_title('x[n]')
axs2[0].set_xlabel('n')
axs2[0].set_ylabel('amplitudine')
axs2[0].grid(True)

# y[n]
axs2[1].stem(n, y_n, basefmt=' ')
axs2[1].set_title('y[n]')
axs2[1].set_xlabel('n')
axs2[1].set_ylabel('amplitudine')
axs2[1].grid(True)

# z[n]
axs2[2].stem(n, z_n, basefmt=' ')
axs2[2].set_title('z[n]')
axs2[2].set_xlabel('n')
axs2[2].set_ylabel('amplitudine')
axs2[2].grid(True)

fig2.tight_layout()
fig2.savefig('semnale_discrete.pdf', dpi=300)  # 💾 salveaza PDF
plt.show()                # afiseaza fereastra cu figura 2

# === 8) Bonus: suprapunere puncte discrete peste curba continua pentru x ===
fig3 = plt.figure(figsize=(8, 3))
plt.plot(t, x_t, label='x(t) continuu')                 # linie continua
plt.stem(t_n, x_n, linefmt='C1-', markerfmt='C1o',
         basefmt=' ', label='x[n] (puncte)', use_line_collection=True)  # puncte + bețe
plt.title('x(t) cu suprapunere puncte x[n]')
plt.xlabel('t [s]')
plt.ylabel('amplitudine')
plt.xlim([0, 0.03])
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('suprapunere_x_continuu_discret.pdf', dpi=300)  # 💾 salveaza PDF
plt.show()
