import numpy as np


# Proximity operator for function F, where


#             F(B) := sum_i (y_i - <A_i, B>)^2

#           A - assumed to be 3-way tensor
#           y - vector of observations

#           Author:    Damian Brzyski
#           Date:      April 24, 2018

# it solves problem


# argmin_B {  sum_i (y_i - <A_i, B>)^2 + delta || B - Dk - Wk/delta ||_F^2  }

###################


def prox_fsvd(y, SVDAx, D_k, Z_k, delta):

    p = Z_k.shape[0]

    AU = SVDAx["U"]
    ASdiag = SVDAx["Sdiag"]
    AVt = SVDAx["Vt"]
    idxs_F = SVDAx["idxs"]

    Ddelta = D_k + Z_k / delta
    DdeltaVec = Ddelta.reshape(p ** 2, -1)
    DdeltaVecU = DdeltaVec[idxs_F]
    Mdelta = AU.T @ DdeltaVecU.reshape(-1) *ASdiag
    ySubs = AVt.T @ Mdelta
    ydelta = y - ySubs

    ##Ridge solution from SVD
    Avty = AVt @ ydelta
    MAVty = (Avty * ASdiag)/ (ASdiag**2 + 2*delta)
    BridgeVec = AU @ MAVty

    # Reconstruct the symmetric matrix
    Bridge = np.zeros(p**2)
    Bridge[idxs_F] = BridgeVec
    Bridge = Bridge.reshape(p,p)
    Bridge = (Bridge + Bridge.T)

    Bnew = Bridge + Ddelta

    return Bnew

if __name__ == "__main__":
# To po prostu odpalaj zaznaczając bo inaczej nie widzi teoretycznie tych zmiennych więc jak puścisz play to wywali bo
# nie widzi części zmiennych. żeby zadziałało kiedy zaznaczasz to muszisz załadować dane ze spinner.py
    delta = 100

    np.random.seed(2021)
    D_k = np.random.randn(5,5)
    D_k = (D_k + D_k.T)/2
    D_k = D_k - np.diag(np.diag(D_k))
    D_k

    np.random.seed(2020)
    Z_k = np.random.randn(5,5)
    Z_k = (Z_k + Z_k.T)/2
    Z_k = Z_k - np.diag(np.diag(Z_k))
    Z_k

    prox_fsvd(y, SVDAx, D_k, Z_k, delta)