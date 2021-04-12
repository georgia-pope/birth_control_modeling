import numpy as np 
import matplotlib.pyplot as plt 
import config as c 
import sympy as sp 

p = c.Params()

# E_2 = p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose
# P_4 = p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose
# P_app = ((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu)))
# InhA = p.h_0 + p.h_1*DomF + p.h_2*Lut_2 + p.h_3*Lut_3

def create_matrix(e_dose, p_dose):
    RP_LH, LH, RP_FSH, FSH, RcF, GrF, DomF, Sc_1, Sc_2, Lut_1, Lut_2, Lut_3, Lut_4 = sp.symbols('RP_LH LH RP_FSH FSH RcF GrF DomF Sc_1 Sc_2 Lut_1 Lut_2 Lut_3 Lut_4')
    print(f'edose before: {p.e_dose}')
    p.e_dose = e_dose
    print(f'edose after: {p.e_dose}')
    p.p_dose = p_dose
    # Pituitary and hypothaalamus axis
    matrix = sp.Matrix([
        # RP_LH
        ((p.V_0LH + (p.V_1LH*((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**8)/(p.K_mLH**8 + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**8)))/(1+((((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu))))/p.K_iLHP))) - \
        ((p.k_LH*(1 + p.c_LHP*(((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu)))))*RP_LH)/(1+p.c_LHE*(p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose))),
        
        # LH
        (p.k_LH*(1+p.c_LHP*(((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu)))))*RP_LH)/(p.v*(1+p.c_LHE*(p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose))) - p.a_LH*LH,

        # RP_FSH
        (p.v_FSH/(1+(p.h_0 + p.h_1*DomF + p.h_2*Lut_2 + p.h_3*Lut_3)/p.K_iFSH_InhA)) - \
        ((p.k_FSH*(1+p.c_FSHP*(((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu)))))*RP_FSH)/(1+p.c_FSHE*(p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**2)),

        # FSH
        (p.k_FSH*(1+p.c_FSHP*(((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu)))))*RP_FSH)/(p.v*(1+p.c_FSHE*(p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**2)) - p.a_FSHE*FSH,

        # Ovarian axis
        # RcF
        (p.b+p.c_1*RcF)*FSH/((1+((((p.p_0 + p.p_1*Lut_3 + p.p_2*Lut_4 + p.p_dose)/2)*(1 + ((p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu / (p.K_mPapp**p.mu + (p.e_0 + p.e_1*GrF + p.e_2*DomF + p.e_3*Lut_4 + p.e_dose)**p.mu))))/p.K_iRcFP))**p.xi) - p.c_2 * (LH**p.alpha) * RcF,

        # GrF
        p.c_2*(LH**p.alpha)*RcF - p.c_3*LH*GrF,

        # DomF
        p.c_3*LH*GrF - p.c_4*(LH**p.gamma)*DomF,

        # Sc_1 
        p.c_4*LH**p.gamma*DomF - p.d_1*Sc_1,

        # Sc_2
        p.d_1*Sc_1 - p.d_2*Sc_2,

        # Lut_1
        p.d_2*Sc_2 - p.k_1*Lut_1,

        # Lut 2
        p.k_1*Lut_1 - p.k_2*Lut_2,

        # Lut_3
        p.k_2*Lut_2 - p.k_3*Lut_3,

        # Lut_4
        p.k_3*Lut_3 - p.k_4*Lut_4
    ])

    variables = [RP_LH, LH, RP_FSH, FSH, RcF, GrF, DomF, Sc_1, Sc_2, Lut_1, Lut_2, Lut_3, Lut_4]

    return variables, matrix