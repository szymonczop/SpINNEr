from sol_options import Options
import numpy as np
from prox_Fsvd import prox_fsvd
from proxH_lasso import calc_proxH_lasso
#this function solves the problem
#
# argmin_B {  0.5*sum_i (y_i - <A_i, B>)^2 + lambda_L || B ||_1  }
#
# y and AA are after regressing X out


def calcluate_spinner_lasso(y, SVDAx, lambdaL, WGTs):
    # Objects
    p0 = SVDAx["U"].shape[0]
    p = int((1 + np.sqrt(1 + 8 * p0)) / 2)

    ## Solver options
    deltaInitial  = Options.deltaInitial2
    mu            = Options.mu
    deltaInc      = Options.deltaInc
    deltaDecr     = Options.deltaDecr
    maxIters      = Options.maxIters
    epsPri        = Options.epsPri
    epsDual       = Options.epsDual

    ## SVD
    U = SVDAx["U"]
    Sdiag = SVDAx["Sdiag"]
    Vt = SVDAx["Vt"]
    idxs = SVDAx["idxs"]

    ## Initial primal and dual matrix
    Dk = np.zeros((p,p))
    Wk = np.zeros((p,p))

    ## ADMM LOOP
    delta = deltaInitial
    counterr = 0

    while True:
        print(counterr)
        Bnew = prox_fsvd(y,SVDAx,Dk,Wk,delta)
        Dnew = calc_proxH_lasso(Bnew,delta,Wk,lambdaL,WGTs)

        Wk = Wk + delta * (Dnew - Bnew)
        rk = Dnew - Bnew
        sk = Dnew - Dk
        rknorm = np.linalg.norm(rk, 'fro')
        Bnorm = np.linalg.norm(Bnew, 'fro')
        rknormR = rknorm / Bnorm
        sknorm = np.linalg.norm(sk, 'fro')
        sknormR = sknorm / np.linalg.norm(Dk, 'fro')
        Dk = Dnew
        counterr += 1

        if counterr % 10 == 0:
            if rknorm > mu * sknorm:
                delta = deltaInc * delta
            else:
                if sknorm > mu * rknorm:
                    delta = delta /deltaDecr

        elif ((rknorm < epsPri) and (sknormR<epsDual)):
            break
        elif counterr > maxIters:
            break
        elif Bnorm < 1e-16:
            Bnew = np.zeros((p,p))
            Dnew = np.zeros((p,p))
            break
    ## optimal value
    DlastVec = Dnew.reshape(p**2,1)
    DlastVecU = DlastVec[idxs]
    MDlast = (U.T @ DlastVecU).reshape((-1,)) * Sdiag
    #eigenvalues_desc_order = np.sort(np.linalg.svd(Dnew)[1])
    optVal = 0.5 * np.linalg.norm(y - Vt.T @ MDlast)**2 + lambdaL * np.sum(abs(Dnew))

    ## outputs

    # Outputs
    out = {}
    out[optVal] = optVal
    out["count"] = counterr
    out["delta"] = delta
    out["Blast"] = Bnew
    out["Dlast"] = Dnew
    out["B"] = Dnew

    return out



#calcluate_spinner_lasso(ytilde, SVDAx, lambdaL, W)