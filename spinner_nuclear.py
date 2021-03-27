import numpy as np
from sol_options import Options
from prox_Fsvd import prox_fsvd
from ProxG import prox_G

# this function solves the problem
#
# argmin_B {  0.5*sum_i (y_i - <A_i, B>)^2 + lambda_N || B ||_*  }
#
# y and AA are after regressing X out

def calclulate_spinner_nuclear(y, SVDAx, lambdaN):
    # cases
    p0 = SVDAx["U"].shape[0]
    p = int((1 + np.sqrt(1 + 8 * p0)) / 2)

    # Solver options

    deltaInitial = Options.deltaInitial1
    mu = Options.mu
    deltaInc = Options.deltaInc
    deltaDecr = Options.deltaDecr
    maxIters = Options.maxIters
    epsPri = Options.epsPri
    epsDual = Options.epsDual

    ## SVD
    U = SVDAx["U"]
    Sdiag = SVDAx["Sdiag"]
    Vt = SVDAx["Vt"]
    idxs = SVDAx["idxs"]

    ## Initial primal and dual matrix
    Ck = np.zeros((p, p))
    Wk = np.zeros((p, p))

    ## ADMM loop
    delta = deltaInitial
    counterr = 0

    ## stoping criteria

    while True:
        print(counterr)
        Bnew = prox_fsvd(y, SVDAx, Ck, Wk, delta)
        Cnew = prox_G(Bnew, -Wk, delta, lambdaN)
        Wk = Wk + delta * (Cnew - Bnew)
        rk = Cnew - Bnew
        sk = Cnew - Ck
        rknorm = np.linalg.norm(rk, 'fro')
        Bnorm = np.linalg.norm(Bnew, 'fro')
        rknormR = rknorm / Bnorm
        sknorm = np.linalg.norm(sk, 'fro')

        sknormR = sknorm / np.linalg.norm(Ck, 'fro')
        Ck = Cnew
        counterr = counterr + 1

        if counterr > 10:
            if rknorm > mu * sknorm:
                delta = deltaInc * delta
            else:
                if sknorm > mu * rknorm:
                    delta = delta / deltaDecr

        if ((rknormR < epsPri) and (sknormR < epsDual)):
            break

        if counterr > maxIters:
            break

        if Bnorm < 1e-16:
            Bnew = np.zeros((p, p))
            Cnew = np.zeros((p, p))
            break

    ##Optimial value
    ClastVec = Cnew.reshape(p ** 2, 1)
    ClastVecU = ClastVec[idxs]
    MClast = (U.T @ ClastVecU).reshape((-1,)) * Sdiag
    eigenvalues_desc_order = np.sort(np.linalg.svd(Cnew)[1])
    optVal = 0.5 * np.linalg.norm(y - Vt.T @ MClast) ** 2 + lambdaN * sum(eigenvalues_desc_order)

    # Outputs
    out = {}
    out[optVal] = optVal
    out["count"] = counterr
    out["delta"] = delta
    out["Blast"] = Bnew
    out["Clast"] = Cnew
    out["B"] = Cnew

    return out

#calclulate_spinner_nuclear(y,SVDAx,lambdaN)
#
# np.sort(np.linalg.svd(X_m)[1])
#
#
#
#
#
# X_m = np.array([[1,2,3],
#                 [0.1,0.2,12],
#                 [1,2,3]])
