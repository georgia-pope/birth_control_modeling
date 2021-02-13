import model
import config as c 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 


days_on_list = [21,22,23,24,25,26,27,28]
days_on_list = np.linspace(21,28, 20)

densities = []
frequencies = []

for days_on in days_on_list:
    tt, yy = model.solve(days_on)
    # plt.plot(tt,yy[1])
    yy = yy[:,1] - yy[:,1].mean()
    frequency, power_spectrum = signal.welch(yy)
    arg_max = power_spectrum.argmax()
    max_density = power_spectrum[arg_max]
    max_frequency = frequency[arg_max]
    # plt.plot(tt,yy)
    # plt.show()
    print(max_density)
    print(max_frequency)
    densities.append(max_density)
    frequencies.append(max_frequency)


plt.plot(days_on_list, densities)
plt.ylabel('Max density')
plt.xlabel('Days on')
plt.title('Max Power Spectral Density vs. Days On')
plt.show()

plt.figure()
plt.plot(days_on_list, frequencies)
plt.ylabel('Max frequency')
plt.xlabel('Days on')
plt.title('Max PSD Frequency vs. Days On')
plt.show()
    


    