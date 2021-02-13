import model
import config as c 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

lambda_up_list = [2,1]
lambda_down_list = [2,1]


# print(f'PSD array shape: {PSD_array.shape}')

def calc_PSD_array(lambda_up_list, lambda_down_list):
    PSD_array = np.zeros((len(lambda_up_list), len(lambda_down_list)))
    for i in range(len(lambda_up_list)):
        for j in range(len(lambda_down_list)):
            lambda_up = lambda_up_list[i]
            lambda_down = lambda_down_list[j]
            tt, yy = model.solve(lam_up=lambda_up, lam_down=lambda_down)
            yy = yy[:,1] - yy[:,1].mean()
            frequency, power_spectrum = signal.welch(yy)
            print(power_spectrum.shape)
            arg_max = power_spectrum.argmax()
            max_density = power_spectrum[arg_max]
            # max_frequency = frequency[arg_max]
            # plt.plot(tt,yy)
            # plt.show()
            print(max_density)
            # print(max_frequency)
            PSD_array[i,j] = max_density
            # frequencies.append(max_frequency)
    return PSD_array



