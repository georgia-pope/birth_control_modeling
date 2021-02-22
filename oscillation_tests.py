import model
import numpy as np
import matplotlib.pyplot as plt
import config as c
from scipy import signal
from scipy.signal import find_peaks

def PSD(yy):
    frequency, power_spectrum = signal.welch(yy)
    arg_max = power_spectrum.argmax()
    max_density = power_spectrum[arg_max]
    max_frequency = frequency[arg_max]
    return max_density, max_frequency

def calc_min_max(yy):
    peaks,_ = find_peaks(yy, distance=1000)
    yy_peaks = yy[peaks]
    min_index = peaks[np.argmin(yy_peaks)]
    max_index = peaks[np.argmax(yy_peaks)]
    min_peak = yy[min_index]
    max_peak = yy[max_index]
    return min_peak, max_peak, min_index, max_index

def calc_avg_value(yy):
    return np.mean(yy)

def test_oscillation(test_type, ind_var, ind_var_vals, func_form=c.func_form):
    """
        inputs:
            test_type = dependent variable
            ind_var = variable we will be changing
            ind_var_vals = values that we will be testing over
        output:

    """
    kwargs = {
        "days_on":c.days_on,
        "lam_up":c.lam_up,
        "lam_down":c.lam_down,
        "func_form":func_form
    }
    densities = []
    frequencies = []
    max_vals = []
    min_vals = []
    decay_ratio = []
    yy_list = []
    for val in ind_var_vals:
        print(f'{ind_var} = {val}')
        kwargs[ind_var] = val
        tt, yy = model.solve(**kwargs)
        tt = tt[10000:]
        yy = yy[10000:,1]
        max_density, max_frequency = PSD(yy)
        densities.append(max_density)
        frequencies.append(max_frequency)

        min_peak, max_peak, min_index, max_index = calc_min_max(yy)
        min_vals.append(min_peak)
        max_vals.append(max_peak)
        decay_ratio.append(max_peak/min_peak)
        yy_list.append((yy, min_index, max_index))


    return tt, yy_list, [densities, frequencies, max_vals, min_vals, decay_ratio]

if False:
    vals = np.linspace(23,25,15)
    ind_var = "days_on"
    tt, yy_list, y_values = test_oscillation('PSD', ind_var, vals)
    y_names = ["Max PSD", "Max Frequency", "Max Amplitude", "Min Amplitude", "Decay Ratio"]

    for i in range(len(yy_list)):
        yy, min_index, max_index = yy_list[i]
        val_label = vals[i]
        plt.figure()
        plt.plot(tt,yy)
        dosing = model.calc_e_dose(np.asarray(tt), 1, c.func_form, lam_up=c.lam_up, lam_down = vals[i])
        title, ylabel, ylim = model.get_plotting_params("LH")
        plt.plot(tt[min_index], yy[min_index], 'o', label = 'min')
        plt.plot(tt[max_index], yy[max_index], 'o', label = 'max')
        plt.plot(tt,dosing*20 + 20, color ='red')
        plt.legend()
        plt.ylim((0,ylim))
        plt.title(f"LH: {ind_var} = {val_label}")
        plt.xlabel("time (days)")

    for i in range(len(y_values)):
        measurement = y_values[i]
        name = y_names[i]
        if name == "Max Amplitude":
            plt.figure()
            plt.plot(vals, measurement, label = 'Max Amplitude')
            plt.plot(vals, y_values[i+1], label = 'Min Amplitude')
            plt.title("Max Amplitude and Min Amplitude")
            plt.ylabel("Amplitude")
            plt.xlabel(ind_var)
        elif name == "Min Amplitude":
            pass
        else:
            plt.figure()
            plt.plot(vals, measurement)
            plt.title(name)
            plt.ylabel(name)
            plt.xlabel(ind_var)
    plt.show()

def lambda_integral(lam_down):
    integral = -(1/lam_down)*(np.exp(-7*lam_down) - 1)
    return integral

def days_on_integral(days_on):
    return days_on - 21

if True:
    days_on_vals = np.linspace(23,25,20)
    days_on_ind_var = "days_on"

    lam_vals = np.linspace(0.25,0.4,20)
    lam_ind_var = "lam_down"

    days_on_area = days_on_integral(days_on_vals)
    lam_area = lambda_integral(lam_vals)
    
    days_on_tt, days_on_yy_list, days_on_y_values = test_oscillation('PSD', days_on_ind_var, days_on_vals, "step_func")
    lam_tt, lam_yy_list, lam_y_values = test_oscillation('PSD', lam_ind_var, lam_vals,"exp_decay")

    plt.plot(days_on_area, days_on_y_values[-1], label='days_on')
    plt.plot(lam_area, lam_y_values[-1], label='lambda down')
    plt.legend()
    plt.title('Decay Ratio vs. Area Under Dose Curve')
    plt.xlabel('Area under dose curve')
    plt.ylabel('Decay ratio')
    plt.show()
    
    


