import model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as c
from oscillation_tests import (PSD, calc_min_max) 
import pandas as pd
import shutil

def get_timeseries(ind_var1, ind_var2, vals1, vals2, func_form=c.func_form, dosing = c.dosing, ind_var3 = None, vals3 = None):
    kwargs = {
        "days_on":c.days_on,
        "lam_up":c.lam_up,
        "lam_down":c.lam_down,
        "func_form":func_form,
        "e_dose":None,
        "p_dose":None,
        "days_missed":c.days_missed,
        "missed_start":c.missed_start
    }
    yy_array = np.zeros((len(vals1)*len(vals2),int(c.num_samples/2)))
    i = 0
    for val1 in vals1:
        j = 0
        for val2 in vals2:
            print(f'{ind_var1} = {val1}')
            print(f'{ind_var2} = {val2}')
            kwargs[ind_var1] = val1
            kwargs[ind_var2] = val2 
            if ind_var3 is not None:
                print('here')
                kwargs[ind_var3] = vals3[j]

            tt, yy = model.solve(**kwargs)

            # Getting the time series after the initial conditions have worn off and only getting LH (y-index 1)
            tt = tt[int(c.num_samples/2):]
            yy = yy[int(c.num_samples/2):,1]
            yy_array[i] = yy
            i += 1
            j+=1
    np.save(f'timeseries/{ind_var1}_{ind_var2}_lam_1.npy', yy_array)
    # import module

    shutil.copyfile('config.py',f'configs/{ind_var1}_{ind_var2}_lam_1.txt')

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

def get_heatmap(plot_vals, ind_var1, ind_var2, vals1, vals2, title, log=False, x_label=None, y_label=None):
    if x_label != None:
        ind_var2 = x_label
    if y_label != None:
        ind_var1 = y_label
    if log:
        plot_vals = np.log(plot_vals)

    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
            'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':24,
            'ytick.labelsize':24}
    pylab.rcParams.update(params)
    plot_vals_df = pd.DataFrame(plot_vals, columns = vals2.round(decimals=1), index = vals1.round(decimals=1))
    plot_vals_df = plot_vals_df.iloc[::-1]
    # sns.color_palette("crest", as_cmap=True)
    plt.figure(figsize=(10,8))
    
    sns.heatmap(plot_vals_df, cmap = "RdBu_r", xticklabels=2, yticklabels=2)
    # sns.heatmap(plot_vals_df, cmap = "RdYlBu")
    plt.ylabel(ind_var1, fontsize=28)
    plt.xlabel(ind_var2, fontsize=28)
    plt.title(title, fontsize=35)
    plt.xticks(rotation=-45)
    plt.yticks(rotation=0)
    plt.show()
    

def main():
    get_timeseries(
        c.ind_var1, 
        c.ind_var2, 
        c.vals1, 
        c.vals2, 
        func_form = c.func_form, 
        ind_var3 = c.ind_var3, 
        vals3 = c.vals3
        )

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