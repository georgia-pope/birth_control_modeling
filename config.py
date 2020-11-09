# Dosing schemes
dosing_dict = {
    'High P_4': [1.3,0],
    'Low P_4': [0.6,0],
    'High E_2': [0,92],
    'Low E_2': [0,40],
    'Low Both': [0.6, 40],
    'Normal Cycle': [0,0]
}

# Select dosing scheme from dosing_dict
dosing = 'Low Both'
on_off = True

# linspace params
num_samples = 20000
upper_bound = 180

# Initial conditions from earlier paper
initial_conditions = [29.65, 6.86, 8.47, 6.15, 3.83, 11.51, 5.48, 19.27, 45.64, 100.73, 125.95, 135.84, 168.71]

# Maps variable name string to variable index
variables = {'RP_LH':0, 'LH':1,' RP_FSH':2, 'FSH':3, 'RcF':4, 'GrF':5, 'DomF':6, 'Sc_1':7,'Sc_2':8, 'Lut_1':9, 'Lut_2':10, 'Lut_3':11, 'Lut_4':12}

# Plotting params 
plotting_params = {   # [title, y_label, ylim] 
    'LH': [dosing + ', ' + 'LH', 'LH (UI/L)', 150],
    'FSH':[dosing + ', ' + 'FSH', 'FSH (UI/L)', 30], 
    'E_2':[dosing + ', ' + 'E2', 'E2 (pg/mL)', 300], 
    'P_4':[dosing + ', ' + 'P4', 'P4 (ng/mL)', 20],
    'RcF':[dosing + ', ' + 'Recruiting Follicle', 'mass', 250],
    'GrF':[dosing + ', ' + 'Growing Follicle', 'mass', 250],
    'DomF':[dosing + ', ' + 'Dominant Follicle', 'mass', 250]
    }

# Class with all the model parameters
class Params:
    def __init__(self):
        self.V_0LH = 500
        self.V_1LH = 4500
        self.K_mLH = 175
        self.K_iLHP = 12.2
        self.k_LH = 2.42
        self.c_LHP = 0.26
        self.c_LHE = 0.004
        self.a_LH = 14
        self.v_FSH = 375
        self.tau = 1.5
        self.K_iFSH_InhA = 1.75
        self.k_FSH = 1.9
        self.c_FSHP = 12
        self.c_FSHE = 0.0018
        self.a_FSHE = 8.21
        self.v = 2.5
        self.b = 0.34
        self.K_iRcFP = 1
        self.xi = 2.2
        self.c_1 = 0.25
        self.c_2 = 0.07
        self.c_3 = 0.027
        self.c_4 = 0.51
        self.d_1 = 0.5
        self.d_2 = 0.56
        self.k_1 = 0.69
        self.k_2 = 0.86
        self.k_3 = 0.85
        self.k_4 = 0.85
        self.alpha = 0.79
        self.gamma = 0.02
        self.e_0 = 30
        self.e_1 = 0.3
        self.e_2 = 0.8
        self.e_3 = 1.67
        self.p_0 = 0.8
        self.p_1 = 0.15
        self.p_2 = 0.13
        self.K_mPapp = 75
        self.mu = 8 
        self.h_0 = 0.4
        self.h_1 = 0.009
        self.h_2 = 0.029
        self.h_3 = 0.018
        self.p_dose = dosing_dict[dosing][0] # 0 or 0.6 or 1.3
        self.e_dose = dosing_dict[dosing][1] # 0 or 40 or 92

