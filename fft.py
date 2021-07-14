import numpy as np
import matplotlib.pyplot as plt


dt = 0.001
Fs = 1/dt # sampling freq
t_end = 1
len_data = int(t_end/dt + 1) # window size
t = np.arange(len_data)*dt
n = np.arange(len_data)
T = len_data/Fs
freq = n/T
sin_20Hz = np.array([])
sin_45Hz = np.array([])
fn_sin = np.array([])
for i in range(0, len_data):
    sin_20Hz = np.append(sin_20Hz, np.sin(2.0 * np.pi * 20.0 * dt * i))
    sin_45Hz = np.append(sin_45Hz, np.sin(2.0 * np.pi * 45.0 * dt * i))

fn_sin = sin_45Hz + sin_20Hz
Y_raw = np.fft.fft(fn_sin)/len_data

y = np.fft.ifft(Y_raw)*len_data
plt.figure()
plt.plot(t, fn_sin)
plt.axis([0, 1, -2, 2])
plt.grid(True)
plt.xlabel('Second [s]')
plt.ylabel('Amplitude [m]')
plt.figure()
plt.plot(t, y)
plt.axis([0, 1, -2, 2])
plt.grid(True)
plt.xlabel('Second [s]')
plt.ylabel('Amplitude [m]')
plt.figure()
plt.stem(freq, np.abs(Y_raw)*2)
plt.axis([0, 100, 0, 2])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [m]')
plt.grid(True)
plt.show()
