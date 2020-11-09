import numpy as np
import scipy.integrate as integrate
from ddeint import ddeint
import matplotlib.pyplot as plt

import config as c 

def calc_e_dose(t, e_dose, on_off = c.on_off, off_set=0):
    """
        Params:
            t (either float or np.array)
            e_dose (float)  
            on_off (bool) True if doing 21 days on 7 days off dosing 
            off_set (int) Can be used to change when on/off dose starts
    """
    t = t+off_set
    if on_off:
        if isinstance(t, np.ndarray):
            e_dose = np.where((t%28) >= 21, 0, e_dose)
        else:
            if (t%28) >= 21:
                e_dose=0
    return e_dose

def calc_p_dose(t, p_dose, on_off = c.on_off, off_set=0):
    """
        Params:
            t (either float or np.array)
            p_dose (float)  
            on_off (bool) True if doing 21 days on 7 days off dosing 
            off_set (int) Can be used to change when on/off dose starts
    """
    t = t+off_set
    if on_off:
        if isinstance(t, np.ndarray):
            p_dose = np.where((t%28) >= 21, 0, p_dose)
        else:
            if (t%28) >= 21:
                p_dose=0
    return p_dose

def derivs(state,t,p):
    RP_LH, LH, RP_FSH, FSH, RcF, GrF, DomF, Sc_1, Sc_2, Lut_1, Lut_2, Lut_3, Lut_4 = state(t)
    RP_LHd, LHd, RP_FSHd, FSHd, RcFd, GrFd, DomFd, Sc_1d, Sc_2d, Lut_1d, Lut_2d, Lut_3d, Lut_4d = state(t-p.tau)
    e_dose = calc_e_dose(t, p.e_dose, c.on_off)
    p_dose = calc_p_dose(t, p.p_dose, c.on_off)
    E_2 = p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + e_dose
    P_4 = p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p_dose
    P_app = (P_4/2)*(1 + (E_2**p.mu / (p.K_mPapp**p.mu + E_2**p.mu)))
    InhA = p.h_0 + p.h_1*DomF + p.h_2*Lut_2 + p.h_3*Lut_3
    InhAd = p.h_0 + p.h_1*DomFd + p.h_2*Lut_2d + p.h_3*Lut_3d
    
    # Pituitary and hypothalamus axis
    synth_LH = (p.V_0LH + (p.V_1LH*(E_2**8)/(p.K_mLH**8 + E_2**8)))/(1+(P_app/p.K_iLHP))
    rel_LH = (p.k_LH*(1 + p.c_LHP*P_app)*RP_LH)/(1+p.c_LHE*E_2)
    delta_RP_LH = synth_LH - rel_LH
    
    delta_LH = (p.k_LH*(1+p.c_LHP*P_app)*RP_LH)/(p.v*(1+p.c_LHE*E_2)) - p.a_LH*LH
    
    synth_FSH = p.v_FSH/(1+InhAd/p.K_iFSH_InhA)
    rel_FSH = (p.k_FSH*(1+p.c_FSHP*P_app)*RP_FSH)/(1+p.c_FSHE*E_2**2)
    delta_RP_FSH = synth_FSH - rel_FSH
    
    delta_FSH = (p.k_FSH*(1+p.c_FSHP*P_app)*RP_FSH)/(p.v*(1+p.c_FSHE*E_2**2)) - p.a_FSHE*FSH
    
    # Ovarian axis
    delta_RcF = (p.b+p.c_1*RcF)*FSH/((1+(P_app/p.K_iRcFP))**p.xi) - p.c_2 * (LH**p.alpha) * RcF
    delta_GrF = p.c_2*(LH**p.alpha)*RcF - p.c_3*LH*GrF
    delta_DomF = p.c_3*LH*GrF - p.c_4*(LH**p.gamma)*DomF
    delta_Sc_1 = p.c_4*LH**p.gamma*DomF - p.d_1*Sc_1
    delta_Sc_2 = p.d_1*Sc_1 - p.d_2*Sc_2
    delta_Lut_1 = p.d_2*Sc_2 - p.k_1*Lut_1
    delta_Lut_2 = p.k_1*Lut_1 - p.k_2*Lut_2
    delta_Lut_3 = p.k_2*Lut_2 - p.k_3*Lut_3
    delta_Lut_4 = p.k_3*Lut_3 - p.k_4*Lut_4
    
    return [delta_RP_LH, delta_LH, delta_RP_FSH, delta_FSH, delta_RcF, delta_GrF, delta_DomF, delta_Sc_1, delta_Sc_2, delta_Lut_1, delta_Lut_2, delta_Lut_3, delta_Lut_4]

def initial_conditions(t):
    return c.initial_conditions

def solve():
    params = c.Params()
    tt = np.linspace(0, c.upper_bound, c.num_samples)
    yy = ddeint(derivs, initial_conditions, tt, fargs=(params,))
    return tt, yy 

def calculate_E2(p, yy, tt):
    e_dose = calc_e_dose(tt, p.e_dose)
    E2 = p.e_0 + p.e_1*yy[:,c.variables['GrF']] + p.e_2*yy[:,c.variables['DomF']] + p.e_3*yy[:,c.variables['Lut_4']] + e_dose
    return E2

def calculate_P4(p, yy, tt):
    P_4 = p.p_0 + p.p_1*yy[:,c.variables['Lut_3']] + p.p_2*yy[:,c.variables['Lut_4']] + calc_p_dose(tt, p.p_dose)
    return P_4

def get_variable(var_name, yy, tt, p):
    if var_name == 'E_2':
        yy = calculate_E2(p, yy, tt)
    elif var_name == 'P_4':
        yy = calculate_P4(p,yy,tt)
    elif var_name in list(c.variables.keys()):
        var_index = c.variables[var_name]
        yy = yy[:, var_index]
    else:
        print('Invalid variable name')
    return yy

def get_plotting_params(var_name):
    if var_name in list(c.plotting_params.keys()):
        title, y_label, ylim = c.plotting_params[var_name]
        return title, y_label, ylim
    else:
        print('Invalid variable name, add variable to plotting params in config file')
    

def plot_single_var(yy, var_name, tt, num_samples = c.num_samples, on_off = c.on_off):
    # Removes first three cycles 
    tt_mask = np.where(tt >= 84, True, False)
    tt = tt[tt_mask]
    yy = yy[tt_mask]

    yy = get_variable(var_name, yy, tt, c.Params())

    title, ylabel, ylim = get_plotting_params(var_name)

    if on_off:
        on_off_mask = np.where((tt%28)>=21, True, False)
        on_tt = tt[~on_off_mask]
        off_tt = tt[on_off_mask]
        on_yy = yy[~on_off_mask]
        off_yy = yy[on_off_mask]
        plt.plot(on_tt,on_yy, '.')
        plt.plot(off_tt,off_yy, '.')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('time (days)')
        plt.ylim((0,ylim))
        plt.show()

    else:
        plt.plot(tt,yy)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('time (days)')
        plt.ylim((0,ylim))
        plt.show()

def plot_four_variables(yy, var_names, tt, num_samples = c.num_samples, on_off = c.on_off):
    # Removes first 3 cycles
    tt_mask = np.where(tt >= 84, True, False)
    tt = tt[tt_mask]
    yy = yy[tt_mask]

    if on_off:
        on_off_mask = np.where((tt%28)>=21, True, False)
        on_tt = tt[~on_off_mask]
        off_tt = tt[on_off_mask]

    y = []
    titles = []
    ylabels = []
    ylims = []
    for var in var_names:
        y.append(get_variable(var, yy, tt, c.Params()))
        title, ylabel, ylim = get_plotting_params(var)
        titles.append(title)
        ylabels.append(ylabel)
        ylims.append(ylim)

    fig, axs = plt.subplots(2, 2)

    if on_off:
        on_yy = y[0][~on_off_mask]
        off_yy = y[0][on_off_mask]
        axs[0, 0].plot(on_tt,on_yy, '.', markersize=1)
        axs[0, 0].plot(off_tt,off_yy, '.', markersize=1)
    else: 
        axs[0, 0].plot(tt, y[0])
    axs[0, 0].set_title(var_names[0])

    if on_off:
        on_yy = y[1][~on_off_mask]
        off_yy = y[1][on_off_mask]
        axs[0, 1].plot(on_tt,on_yy, '.', markersize=1)
        axs[0, 1].plot(off_tt,off_yy, '.', markersize=1)
    else:
        axs[0, 1].plot(tt, y[1])
    axs[0, 1].set_title(var_names[1])

    if on_off:
        on_yy = y[3][~on_off_mask]
        off_yy = y[3][on_off_mask]
        axs[1, 1].plot(on_tt,on_yy, '.', markersize=1)
        axs[1, 1].plot(off_tt,off_yy, '.', markersize=1)
    else:
        axs[1, 1].plot(tt, y[3])
    axs[1, 1].set_title(var_names[3])

    if on_off:
        on_yy = y[2][~on_off_mask]
        off_yy = y[2][on_off_mask]
        axs[1, 0].plot(on_tt,on_yy, '.', markersize=1)
        axs[1, 0].plot(off_tt,off_yy, '.', markersize=1)
    else:
        axs[1, 0].plot(tt, y[2])
    axs[1, 0].set_title(var_names[2])

    i = 0
    for ax in axs.flat:
        ax.set(xlabel='Time (days)', ylabel=ylabels[i], ylim=(0,ylims[i]))
        i += 1

    fig.suptitle('Dose: ' + c.dosing)
    plt.tight_layout()
    plt.show()
    
    

tt, yy = solve()

# var_names = ['E_2', 'P_4', 'FSH', 'LH']
var_names = ['LH', 'RcF', 'GrF', 'DomF']

plot_four_variables(yy, var_names, tt)

# for var in var_names:
#     plot_single_var(yy, var, tt)


