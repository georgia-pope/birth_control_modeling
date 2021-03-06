import numpy as np
import scipy.integrate as integrate
from ddeint import ddeint
import matplotlib.pyplot as plt

import config as c 

def step_func(t, dose, days_on):
    """ 
        Represents exogenous hormone levels as a step function whose value is dose for days_on
        days and 0 for (28 - days_on) days
        Params:
            t (float or ndarray)
            dose (float) dose of hormone that is being represented by a step function
            days_on (float) number of days that birth control pill is taken
        Returns:
            if t is a float, returns a float representing the serum level of the hormone
            if t is an ndarray, returns an ndarray of serum levels of the hormone for the given time period
    """
    if isinstance(t, np.ndarray):
        dose = np.where((t%28) >= days_on, 0, dose)
    else:
        if (t%28) >= days_on:
            dose=0
    return dose

def soft_step(tt, dose, days_on, lam_up = c.lam_up, lam_down=c.lam_down):
    """
        Represents exogenous hormone levels as a function that resembles a step function, however the
        changes in value are described by sigmoidal curves
        Params:
            tt (float or ndarray)
            dose (float) dose of hormone that is being represented by a step function
            days_on (float) number of days that birth control pill is taken
            lam_up (float) lambda value in sigmoidal function representing the increase in exogenous hormones
                after days off
            lam_down (float) lambda value in sigmoidal function representing the decrease in exogenous hormones
                when the days off start
        Returns:
            if tt is a float, returns a float representing the serum level of the hormone
            if tt is an ndarray, returns an ndarray of serum levels of the hormone for the given time period
    """
    # Shifting t values to make calculation easier
    shift = (28 - days_on)/2
    t = tt + shift

    # Shifting curves so that the decrease or increase in hormone levels occurs on the correct day
    # (otherwise the sigmoid would be centered on the correct day)
    up_lambda_shift = np.log((1-0.01)/0.01)/lam_up
    down_lambda_shift = -np.log(0.99/(1-0.99))/lam_down

    if isinstance(tt, np.ndarray):
        soft_step_dose = np.where(
            ((t%28) <= days_on/2 + shift), 
            dose*(1/(1+np.exp(-lam_up*((t%28)-shift-up_lambda_shift)))), 
            - dose*(1/(1+np.exp(-lam_down*((t%28)- days_on - shift + down_lambda_shift)))) + dose
            )
    else:
        if (t%28) <= days_on/2 + shift:
            soft_step_dose = dose*(1/(1+np.exp(-lam_up*((t%28)-shift-up_lambda_shift))))
        else:
            soft_step_dose = - dose*(1/(1+np.exp(-lam_down*((t%28)- days_on - shift + down_lambda_shift)))) + dose
    
    return soft_step_dose

def exp_decay(tt, dose, days_on, lam=1):
    """
        Represents exogenous hormone levels as a function where the hormone level is at dose for days_on days,
        exponentially decays to 0 for (28 - days_on) days, and steps back up to dose 
        Params:
            tt (float or ndarray)
            dose (float) dose of hormone that is being represented by a step function
            days_on (float) number of days that birth control pill is taken
            lam (float) lambda value for exponential decay
        Returns:
            if tt is a float, returns a float representing the serum level of the hormone
            if tt is an ndarray, returns an ndarray of serum levels of the hormone for the given time period
    """
    if isinstance(tt, np.ndarray):
        exp_decay_dose = np.where(
            (tt%28) <= days_on, 
            dose,
            dose*np.exp(-lam*((tt%28)-days_on)) 
            )
    else:
        if (tt%28) <= days_on:
            return dose
        else:
            exp_decay_dose = dose*np.exp(-lam*((tt%28)-days_on)) 
    return exp_decay_dose

def lin_decay(tt, dose, days_on, intercept = 28):
    """
        Represents exogenous hormone levels as a function where the hormone level is at dose for days_on days,
        linearly decays to 0 for (28 - days_on) days with a slope determined by intercept, 
        and steps back up to dose 
        Params:
            tt (float or ndarray)
            dose (float) dose of hormone that is being represented by a step function
            days_on (float) number of days that birth control pill is taken
            intercept (float) x-intercept that determines slope of decaying line 
                intercept = days_on -> step function
                intercept = 28 -> line with slope -dose/(28 - days_on)
        Returns:
            if tt is a float, returns a float representing the serum level of the hormone
            if tt is an ndarray, returns an ndarray of serum levels of the hormone for the given time period
    """
    def calc_decay_line(dose=dose, days_on=days_on, intercept=intercept):
        rel_intercept = intercept - days_on
        return lambda x: -dose*(x-days_on)/rel_intercept + dose 
    decay_line = calc_decay_line()
    if isinstance(tt, np.ndarray):
        lin_decay_dose = np.where(
            ((tt%28) >= days_on) & ((tt%28) <= intercept), 
            decay_line(tt%28), 
            step_func(tt,dose,days_on)
            )
    else:
        if ((tt%28) >= days_on) and ((tt%28) <= intercept):
            return decay_line(tt%28)
        else:
            lin_decay_dose = step_func(tt,dose,days_on) 
    return lin_decay_dose

def missed_days(tt, dose, days_missed = 1, missed_start = 5, double_next = False, lam=1):
    if isinstance(tt, np.ndarray):
        exp_dose = exp_decay(tt, dose, 21, lam=lam)
                    
        dose = np.where(
            (tt%(c.upper_bound/2) <  (missed_start+days_missed)) & (tt%(c.upper_bound/2) > (missed_start)),
            # (tt%28 < (missed_start+days_missed))  &  (tt%28 > (missed_start)), 
            dose*np.exp(-lam*((tt%(c.upper_bound/2))-missed_start)), 
            exp_dose
            )
    else: 
        if tt % 1 == 0:
            print(days_missed)
        # if (tt%28 < (missed_start+days_missed))  &  (tt%28 > (missed_start)):
        if (tt%(c.upper_bound/2) <  (missed_start+days_missed)) & (tt%(c.upper_bound/2) > (missed_start+days_missed)):
            # dose = dose*np.exp(-lam*((tt%28)-missed_start))
            dose = dose*np.exp(-lam*((tt%(c.upper_bound/2))-missed_start))
        else:
            dose = exp_dose = exp_decay(tt, dose, 21, lam=lam)

    return dose

def calc_e_dose(
    t, e_dose, func_form = c.func_form, 
    on_off = c.on_off, off_set=c.off_set, 
    days_on = c.days_on, 
    lam_up=c.lam_up, lam_down = c.lam_down, missed_start = c.missed_start, days_missed = c.days_missed
    ):
    """
        Params:
            t (either float or np.array)
            e_dose (float)  
            func_form (string) indicates which functional form is being used to represent estrogen dosing
            on_off (bool) True if doing on/off dosing 
            off_set (int) Can be used to change when on/off dose starts
            days_on (float) number of days taking hormonal birth control
            lam_up (float) if func_form is "soft_step", lam_up is lam_up param for soft_step()
            lam_down (float) if func_form is "soft_step", lam_down is lam_down param for soft_step()
                if func_form is "exp_decay", lam_down is the lam param for exp_decay()
                if func_form is "lin_decay", lam_down is the intercept param for lin_decay()
        Returns:
            a value or an array of values representing the serum level of exogenous
            estrogen at the given time(s)
    """
    t = t+off_set
    if func_form == "constant":
        if isinstance(t, np.ndarray):
            e_dose = np.full_like(t,e_dose)
        else:
            return e_dose
    elif func_form == "step_func":
        e_dose = step_func(t, e_dose, days_on)
    elif func_form == "soft_step":
        e_dose = soft_step(t, e_dose, days_on, lam_up, lam_down)
    elif func_form == "exp_decay":
        e_dose = exp_decay(t, e_dose, days_on, lam_down)
    elif func_form == "lin_decay":
        e_dose = lin_decay(t, e_dose, days_on, lam_down)
    elif func_form == "missed_days":
        e_dose = missed_days(t, e_dose, missed_start = missed_start, days_missed = days_missed, double_next = False, lam=1)
    return e_dose

def calc_p_dose(
    t, p_dose, func_form = c.func_form, 
    on_off = c.on_off, off_set=c.off_set, 
    days_on = c.days_on, 
    lam_up=c.lam_up, lam_down = c.lam_down, missed_start = c.missed_start, days_missed = c.days_missed
    ):
    """
        Params:
            t (either float or np.array)
            p_dose (float)  
            func_form (string) indicates which functional form is being used to represent estrogen dosing
            on_off (bool) True if doing on/off dosing 
            off_set (int) Can be used to change when on/off dose starts
            days_on (float) number of days taking hormonal birth control
            lam_up (float) if func_form is "soft_step", lam_up is lam_up param for soft_step()
            lam_down (float) if func_form is "soft_step", lam_down is lam_down param for soft_step()
                if func_form is "exp_decay", lam_down is the lam param for exp_decay()
                if func_form is "lin_decay", lam_down is the intercept param for lin_decay()
        Returns:
            a value or an array of values representing the serum level of exogenous progesterone at 
            the given time(s)
    """
    t = t+off_set
    if func_form == "constant":
        if isinstance(t, np.ndarray):
            p_dose = np.full_like(t,p_dose)
        else:
            return p_dose
    elif func_form == "step_func":
        p_dose = step_func(t, p_dose, days_on)
    elif func_form == "soft_step":
        p_dose = soft_step(t, p_dose, days_on, lam_up, lam_down)
    elif func_form == "exp_decay":
        p_dose = exp_decay(t, p_dose, days_on, lam_down)
    elif func_form == "lin_decay":
        p_dose = lin_decay(t, p_dose, days_on, lam_down)
    elif func_form == "missed_days":
        p_dose = missed_days(t, p_dose, missed_start = missed_start, days_missed = days_missed, double_next = False, lam=1)
    return p_dose

def derivs(state, t, p, days_on, lam_up=c.lam_up, lam_down = c.lam_down, func_form=c.func_form,missed_start = c.missed_start, days_missed = c.days_missed):
    """
        Params:
            state
            t (np.array)
            p (dict) parameter dictionary 
            days_on (float) number of days taking hormonal birth control
            lam_up (float) if func_form is "soft_step", lam_up is lam_up param for soft_step()
            lam_down (float) if func_form is "soft_step", lam_down is lam_down param for soft_step()
                if func_form is "exp_decay", lam_down is the lam param for exp_decay()
                if func_form is "lin_decay", lam_down is the intercept param for lin_decay()
            func_form (string) indicates which functional form is being used to represent hormone dosing
    """
    RP_LH, LH, RP_FSH, FSH, RcF, GrF, DomF, Sc_1, Sc_2, Lut_1, Lut_2, Lut_3, Lut_4 = state(t)
    RP_LHd, LHd, RP_FSHd, FSHd, RcFd, GrFd, DomFd, Sc_1d, Sc_2d, Lut_1d, Lut_2d, Lut_3d, Lut_4d = state(t-p.tau)
    e_dose = calc_e_dose(t, p.e_dose, func_form, c.on_off, days_on = days_on, lam_up=lam_up, lam_down = lam_down, missed_start = missed_start, days_missed = days_missed)
    p_dose = calc_p_dose(t, p.p_dose, func_form, c.on_off, days_on = days_on, lam_up=lam_up, lam_down = lam_down, missed_start = missed_start, days_missed = days_missed)
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
    """
        Resurns initial conditions as specified in config.py
    """
    return c.initial_conditions

def solve(days_on=c.days_on, lam_up=c.lam_up, lam_down = c.lam_down, func_form=c.func_form, e_dose=None, p_dose=None, missed_start = c.missed_start, days_missed = c.days_missed):
    """
        Params:
            days_on (float) number of days taking hormonal birth control
            lam_up (float) if func_form is "soft_step", lam_up is lam_up param for soft_step()
            lam_down (float) if func_form is "soft_step", lam_down is lam_down param for soft_step()
                if func_form is "exp_decay", lam_down is the lam param for exp_decay()
                if func_form is "lin_decay", lam_down is the intercept param for lin_decay()
            func_form (string) indicates which functional form is being used to represent hormone dosing
            e_dose (float) default value is None, if e_dose = None, e_dose = c.e_dose
            p_dose (float) default value is None, if p_dose = None, p_dose = c.p_dose
        Returns:
            tt (ndarray) 
            yy (ndarray)
    """
    params = c.Params()
    print(f"missed days: {days_missed}")
    print(f"missed start: {missed_start}")
    if e_dose is not None:
        params.e_dose = e_dose
    if p_dose is not None:
        params.p_dose = p_dose
    tt = np.linspace(0, c.upper_bound, c.num_samples)
    yy = ddeint(derivs, initial_conditions, tt, fargs=(params,days_on,lam_up,lam_down,func_form,missed_start, days_missed))
    return tt, yy 

def calculate_E2(p, yy, tt, days_on, lam_up=c.lam_up, lam_down = c.lam_down, func_form=c.func_form, missed_start = c.missed_start, days_missed = c.days_missed):
    e_dose = calc_e_dose(tt, p.e_dose, func_form, days_on=days_on, lam_up=lam_up, lam_down = lam_down, missed_start = missed_start, days_missed = days_missed)
    E2 = p.e_0 + p.e_1*yy[:,c.variables['GrF']] + p.e_2*yy[:,c.variables['DomF']] + p.e_3*yy[:,c.variables['Lut_4']] + e_dose
    return E2

def calculate_P4(p, yy, tt, days_on, lam_up=c.lam_up, lam_down = c.lam_down, func_form=c.func_form, missed_start = c.missed_start, days_missed = c.days_missed):
    p_dose = calc_p_dose(tt, p.p_dose, func_form, days_on=days_on, lam_up=lam_up, lam_down = lam_down, missed_start = missed_start, days_missed = days_missed)
    P_4 = p.p_0 + p.p_1*yy[:,c.variables['Lut_3']] + p.p_2*yy[:,c.variables['Lut_4']] + p_dose
    return P_4

def get_variable(var_name, yy, tt, p, days_on, lam_up=c.lam_up, lam_down = c.lam_down, func_form=c.func_form, missed_start = c.missed_start, days_missed = c.days_missed):
    """
        Calculates values of var_name for a given timeseries yy (for plotting purposes)
    """
    if var_name == 'E_2':
        yy = calculate_E2(p, yy, tt, days_on, lam_up=lam_up, lam_down = lam_down, func_form=func_form, missed_start = missed_start, days_missed = days_missed)
    elif var_name == 'P_4':
        yy = calculate_P4(p,yy,tt, days_on, lam_up=lam_up, lam_down = lam_down, func_form=func_form, missed_start = missed_start, days_missed = days_missed)
    elif var_name in list(c.variables.keys()):
        var_index = c.variables[var_name]
        yy = yy[:, var_index]
    else:
        print('Invalid variable name')
    return yy

def get_plotting_params(var_name):
    """
        Retrieves plotting parameters from config.py

        Returns:
            title, y_label, ylim
    """
    if var_name in list(c.plotting_params.keys()):
        title, y_label, ylim = c.plotting_params[var_name]
        return title, y_label, ylim
    else:
        print('Invalid variable name, add variable to plotting params in config file')
    

def plot_single_var(yy, var_name, tt, num_samples = c.num_samples, on_off = c.on_off, days_on = c.days_on):
    # Removes first three cycles 
    tt_mask = np.where(tt >= 84, True, False)
    tt = tt[tt_mask]
    yy = yy[tt_mask]

    yy = get_variable(var_name, yy, tt, c.Params(), days_on)

    title, ylabel, ylim = get_plotting_params(var_name)

    if on_off:
        on_off_mask = np.where((tt%28)>=days_on, True, False)
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

def plot_four_variables(yy, var_names, tt, 
num_samples = c.num_samples, on_off = c.on_off, days_on = c.days_on,
lam_up=c.lam_up, lam_down=c.lam_down, func_form=c.func_form
):
    # Removes first 3 cycles
    tt_mask = np.where(tt >= 84, True, False)
    tt = tt[tt_mask]
    yy = yy[tt_mask]

    if on_off:
        on_off_mask = np.where((tt%28)>=days_on, True, False)
        on_tt = tt[~on_off_mask]
        off_tt = tt[on_off_mask]

    y = []
    titles = []
    ylabels = []
    ylims = []
    for var in var_names:
        y.append(get_variable(var, yy, tt, c.Params(), days_on, lam_up, lam_down))
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

def plot_four_variables_red(
    yy, var_names, tt, 
    num_samples = c.num_samples, 
    on_off = c.on_off, days_on = c.days_on,
    lam_up=c.lam_up, lam_down = c.lam_down, func_form=c.func_form, missed_start = c.missed_start, days_missed = c.days_missed
    ):
    # Removes first 3 cycles
    tt_mask = np.where(tt >= 84, True, False)
    tt = tt[tt_mask]
    yy = yy[tt_mask]
    dosing = calc_e_dose(np.asarray(tt), 1, func_form, lam_up=lam_up, lam_down = lam_down, missed_start = missed_start, days_missed = days_missed)
    print(np.asarray(tt))
    print(dosing)
    # print(tt)

    # dosing = np.where((tt%28)>=days_on, 0, 1)

    y = []
    titles = []
    ylabels = []
    ylims = []
    print('\nStarting to get variables')
    for var in var_names:
        print(f'Getting {var}')
        y.append(get_variable(var, yy, tt, c.Params(), days_on,  lam_up=lam_up, lam_down = lam_down, func_form=func_form, missed_start = missed_start, days_missed = days_missed))
        title, ylabel, ylim = get_plotting_params(var)
        titles.append(title)
        ylabels.append(ylabel)
        ylims.append(ylim)

    fig, axs = plt.subplots(2, 2)

    if on_off:
        ax1 = axs[0,0]
        ax2 = ax1.twinx()
        ax1.plot(tt,y[0])
        ax2.plot(tt,dosing, c='r')
        ax2.set(ylim=(-3, 1.5))
    else: 
        axs[0, 0].plot(tt, y[0])
    axs[0, 0].set_title(var_names[0])

    if on_off:
        ax1 = axs[0,1]
        ax2 = ax1.twinx()
        ax1.plot(tt,y[1])
        ax2.plot(tt,dosing, c='r')
        ax2.set(ylim=(-3, 1.5))
    else:
        axs[0, 1].plot(tt, y[1])
    axs[0, 1].set_title(var_names[1])

    if on_off:
        ax1 = axs[1,1]
        ax2 = ax1.twinx()
        ax1.plot(tt,y[3])
        ax2.plot(tt,dosing, c='r')
        ax2.set(ylim=(-3, 1.5))
    else:
        axs[1, 1].plot(tt, y[3])
    axs[1, 1].set_title(var_names[3])

    if on_off:
        ax1 = axs[1,0]
        ax2 = ax1.twinx()
        ax1.plot(tt,y[2])
        ax2.plot(tt,dosing, c='r')
        ax2.set(ylim=(-3, 1.5))
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

if False:
    tt, yy = solve()

    var_names_1 = ['E_2', 'P_4', 'FSH', 'LH']
    var_names_2 = ['LH', 'RcF', 'GrF', 'DomF']

    # # plot_four_variables(yy, var_names, tt, days_on = 21)
    plt.plot(tt, yy[:,1])
    plt.show()
    plot_four_variables_red(yy, var_names_1, tt, days_on = 21)
    plot_four_variables_red(yy, var_names_2, tt, days_on = 21)
    
if False:

    # tt, yy = solve(func_form="missed_days", missed_start= 10, days_missed=1)
    tt = np.linspace(0, c.upper_bound, c.num_samples)
    e_dose = calc_e_dose(tt,1,days_missed=1, missed_start=7)
    plt.plot(tt,e_dose)
    plt.show()

    # var_names = ['E_2', 'P_4', 'FSH', 'LH']
    # plot_four_variables_red(yy, var_names, tt, days_on = 21, func_form="missed_days", missed_start=5, days_missed=0)

  

