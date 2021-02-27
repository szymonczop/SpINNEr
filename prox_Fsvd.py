import numpy as np


# Proximity operator for function F, where


#             F(B) := sum_i (y_i - <A_i, B>)^2

#           A - assumed to be 3-way tensor
#           y - vector of observations

#           Author:    Damian Brzyski
#           Date:      April 24, 2018

# it solves problem


# argmin_B {  sum_i (y_i - <A_i, B>)^2 + delta || B - Dk - Wk/delta ||_F^2  }
def prox_fsvd(y, SVDAx, D_k, Z_k, delta):
    p = Z_k.shape[0]

    AU = SVDAx["U"]
    ASdiag = SVDAx["Sdiag"]
    AVt = SVDAx["Vt"]
    idxs_F = SVDAx["idxs"]

    Ddelta = D_k + Z_k / delta
    DdeltaVec = Ddelta.reshape(p ** 2, -1)
    DdeltaVecU = DdeltaVec[idxs_F]
    # Mdelta = AU.T @ DdeltaVecU * ASdiag
    # #ySubs = AVt.T @ Mdelta
    ySubs = AU @ ASdiag @ AVt @ DdeltaVecU
    ydelta = y - ySubs.reshape(len(y), )

    right_side = AU.T @ ydelta
    middle = np.diagonal(ASdiag) / (np.diagonal(ASdiag) ** 2 + 2 * delta)
    BridgeVec = AVt.T @ (middle * right_side)
    # Ridge solution from SVD
    # AVty       = AVt @ ydelta
    # MAVty      = (AVty*ASdiag)/(ASdiag**2 + 2*delta)# element-wise
    # BridgeVec  = AU * MAVty

    # RECONSTRUCT THE SYMMETRIC MATRIX
    Bridge = np.zeros(p ** 2)
    Bridge[idxs_F] = BridgeVec
    Bridge = Bridge.reshape(p, p)
    Bridge = Bridge + Bridge.T

    # B UPDATE

    Bnew = Bridge + Ddelta  # POWRÓT Z BETA_FALKA DO BETA

    return Bnew


##### TUTAJ SPORO RZECZY JEST ZAŁOŻONYCH Z GÓRY NIE WIEM CZY TAK POWINNO BYć
##### Z_K i D_K są symetryczne z zerami na przekątnej
#if __name__=="__main__":
n = 20
y = np.random.rand(n)
Z_k = np.random.randint(0, 10, size=(p, p))
Z_k = (Z_k + Z_k.T) / 2
np.fill_diagonal(Z_k, 0)
delta = 2
D_k = np.random.randint(0, 10, size=(p, p))
D_k = (D_k + D_k.T) / 2
np.fill_diagonal(D_k, 0)

prox_fsvd(y, SVDAx, D_k, Z_k, delta)
