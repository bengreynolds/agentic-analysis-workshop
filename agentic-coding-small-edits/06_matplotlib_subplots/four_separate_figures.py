import matplotlib.pyplot as plt
import numpy as np


time = np.linspace(0, 10, 100)
signal_a = np.sin(time)
signal_b = np.cos(time)
signal_c = np.sin(time) + np.cos(time)
signal_d = np.sin(time * 2)

plt.figure()
plt.plot(time, signal_a)
plt.title("Signal A")
plt.xlabel("Time")
plt.ylabel("Response")

plt.figure()
plt.plot(time, signal_b)
plt.title("Signal B")
plt.xlabel("Time")
plt.ylabel("Response")

plt.figure()
plt.plot(time, signal_c)
plt.title("Signal C")
plt.xlabel("Time")
plt.ylabel("Response")

plt.figure()
plt.plot(time, signal_d)
plt.title("Signal D")
plt.xlabel("Time")
plt.ylabel("Response")

plt.show()
