import model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as c
from oscillation_tests import (PSD, calc_min_max) 
import pandas as pd

def get_timeseries(ind_var1, ind_var2, vals1, vals2, func_form=c.func_form):
    kwargs = {
        "days_on":c.days_on,
        "lam_up":c.lam_up,
        "lam_down":c.lam_down,
        "func_form":func_form,
        "e_dose":None,
        "p_dose":None
    }
    yy_array = np.zeros((len(vals1)*len(vals2),int(c.num_samples/2)))
    i = 0
    for val1 in vals1:
        for val2 in vals2:
            print(f'{ind_var1} = {val1}')
            print(f'{ind_var2} = {val2}')
            kwargs[ind_var1] = val1
            kwargs[ind_var2] = val2 
            tt, yy = model.solve(**kwargs)

            # Getting the time series after the initial conditions have worn off and only getting LH (y-index 1)
            tt = tt[int(c.num_samples/2):]
            yy = yy[int(c.num_samples/2):,1]
            yy_array[i] = yy
            i += 1
    np.save(f'{ind_var1}_{ind_var2}.npy', yy_array)

def get_plotting_vals(filename, vals1, vals2):
    yy_array = np.load(filename)
    densities = np.zeros(yy_array.shape[0])
    frequencies = np.zeros(yy_array.shape[0])
    decay_ratio = np.zeros(yy_array.shape[0])
    avg_amp = np.zeros(yy_array.shape[0])
    for i in range(yy_array.shape[0]):
        yy = yy_array[i]
        
        max_density, max_frequency = PSD(yy)
        densities[i] = max_density
        frequencies[i] = max_frequency

        min_peak, max_peak, _, _ = calc_min_max(yy)
        decay_ratio[i] = max_peak/min_peak
        avg_amplitude = (min_peak+max_peak)/2
        avg_amp[i] = avg_amplitude - np.min(yy)
        # avg_amp[i] = avg_amplitude


    h = len(vals1)
    w = len(vals2)
    return densities.reshape((h,w)), frequencies.reshape((h,w)), decay_ratio.reshape((h,w)), avg_amp.reshape((h,w))

def get_heatmap(plot_vals, ind_var1, ind_var2, vals1, vals2, title, log=False):
    if log:
        plot_vals = np.log(plot_vals)
    plot_vals_df = pd.DataFrame(plot_vals, columns = vals2, index = vals1)
    plot_vals_df = plot_vals_df.iloc[::-1]
    sns.heatmap(plot_vals_df)
    plt.ylabel(ind_var1)
    plt.xlabel(ind_var2)
    plt.title(title)
    plt.show()
    

def main():
    vals1 = [0.25, 0.5, 1, 2, 3, 4, 10]
    vals2 = [0.25, 0.5, 1, 2, 3, 4, 10]
    ind_var1 = "lam_up"
    ind_var2 = "lam_down"
    get_timeseries(ind_var1, ind_var2, vals1, vals2, func_form = "soft_step")

if __name__ == "__main__":
    main()

# if False:
#     vals1 = np.linspace(21, 28, 8)
#     vals2 = np.linspace(0, 1.3, 6)
#     ind_var1 = "days_on"
#     ind_var2 = "p_dose"
#     get_timeseries(ind_var1, ind_var2, vals1, vals2, func_form = "constant")
#     densities, frequencies, decay_ratio = get_plotting_vals(f"{ind_var1}_{ind_var2}.npy", vals1, vals2)
#     get_heatmap(decay_ratio, ind_var1, ind_var2, vals1, vals2, "Decay Ratio")
#     get_heatmap(densities, ind_var1, ind_var2, vals1, vals2, "Max PSD")
#     get_heatmap(frequencies, ind_var1, ind_var2, vals1, vals2, "Max PSD Frequency")