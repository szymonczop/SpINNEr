import numpy as np

def calc_min_norm_estim(y, SVDAx):
    # objects
    p0 = SVDAx["U"].shape[0]
    p = int((1+np.sqrt(1+8*p0))/2)

    #SVD outputs
    U      = SVDAx["U"]
    Sdiag  = SVDAx["Sdiag"]
    Vt     = SVDAx["Vt"]
    idxs   = SVDAx["idxs"]

    #Estimate
    Vty  = Vt @ y /Sdiag
    Bvec = U@Vty
    Bestim  = np.zeros(p**2)
    Bestim[idxs] = Bvec

    Bestim  = Bestim.reshape(p,p) # p-by-p matrix
    Bestim  = Bestim + Bestim.T

    return Bestim

#calc_min_norm_estim(y, SVDAx)








