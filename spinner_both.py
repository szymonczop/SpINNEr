


# This function solves the problem
#--------------------------------------------------------------------------
#    argmin_{B, beta} {   0.5*sum_i ( y_i - X*beta - <A_i, B> )^2 +
#                                 lambda_N*|| B ||_* +
#                               lambda_L*|| vec(B o W) ||_1    }
#--------------------------------------------------------------------------
# for given pair of positive regularization parameters, lambdaN and lambdaL.
# In order to do so the specific implementation of ADMM algorithm is used.
import numpy as np
from sol_options import Options
from prox_Fsvd import prox_fsvd
from ProxG import prox_G
from prox_h import prox_H

def calculate_spinner_both(y, SVDAx, lambdaN, lambdaL, WGTs):

    # cases
    p0 = SVDAx["U"].shape[0]
    p = int((1 + np.sqrt(1 + 8 * p0)) / 2)

    # Solver options
    deltaInitial1 = Options.deltaInitial1
    deltaInitial2 = Options.deltaInitial2
    scaleStep     = Options.scaleStep
    ratioStep     = Options.ratioStep
    mu            = Options.mu
    deltaInc      = Options.deltaInc
    deltaDecr     = Options.deltaDecr
    ratioInc      = Options.ratioInc
    ratioDecr     = Options.ratioDecr
    maxIters      = Options.maxIters
    epsPri        = Options.epsPri
    epsDual       = Options.epsDual

    ## SVD
    U = SVDAx["U"]
    Sdiag = SVDAx["Sdiag"]
    Vt = SVDAx["Vt"]
    idxs = SVDAx["idxs"]

    # Initial primal and dual matrix
    Dk = np.zeros((p,p))
    W1k = np.zeros((p,p))
    W2k = np.zeros((p,p))

    ## ADMM loop
    delta1    = deltaInitial1
    delta2    = deltaInitial2
    counterr = 0
    CsB      = []
    DsB      = []
    DsDp     = []
    Dlts1    = []
    Dlts2    = []

    while True:
        print(counterr)
        Bnew = prox_fsvd(y, SVDAx, Dk, W1k, delta1)
        Cnew = prox_G(Dk, W2k, delta2, lambdaN)
        Dnew = prox_H(Bnew, Cnew, delta1, delta2, W1k, W2k, lambdaL, WGTs)
        W1k   = W1k + delta1*(Dnew - Bnew)
        W2k   = W2k + delta2*(Dnew - Cnew)

        rk1 = Cnew - Bnew
        rk2 = Dnew - Bnew
        sk = Dnew - Dk
        rknorm1 = np.linalg.norm(rk1, 'fro')
        Bnorm = np.linalg.norm(Bnew, 'fro')
        rknormR1 = rknorm1 / Bnorm
        rknorm2 = np.linalg.norm(rk2, 'fro')
        rknormR2 = rknorm2 / Bnorm
        sknorm = np.linalg.norm(sk, 'fro')
        sknormR = sknorm / np.linalg.norm(Dk, 'fro')
        counterr = counterr + 1
        CsB.append(rknormR1)
        DsB.append(rknormR2)
        DsDp.append(sknormR)
        Dlts1.append(delta1)
        Dlts2.append(delta2)
        Dk = Dnew

        # rations update
        if counterr % 20 == 10:
            if rknorm1 > mu*rknorm2:
                ratioStep = ratioStep*ratioInc
            else:
                if rknorm2 > mu * rknorm1:
                    ratioStep = ratioStep/ratioDecr
        #scale update
        if counterr % 20 == 0:
            if np.mean([rknorm1,rknorm2]) > mu * sknorm:
                scaleStep = scaleStep * deltaInc
            else:
                if sknorm > mu * np.mean([rknorm1,rknorm2]):
                    scaleStep = scaleStep/deltaDecr

        delta1 = scaleStep*deltaInitial1
        delta2 = scaleStep*ratioStep*deltaInitial2

        # stopping cryteria
        if ((rknormR1 < epsPri) and (rknormR2 < epsPri) and (sknormR < epsDual)):
            break
        if counterr > maxIters:
            break
        if Bnorm < 1e-16:
            Bnew = np.zeros((p, p))
            Cnew = np.zeros((p, p))
            Dnew = np.zeros((p, p))
            break

    DlastVec = Dnew.reshape(p**2,1)
    DlastVecU = DlastVec[idxs]
    MDlast = (U.T @ DlastVecU).reshape((-1,)) * Sdiag
    eigenvalues_desc_order = np.sort(np.linalg.svd(Dnew)[1])
    optVal = 0.5 * np.linalg.norm(y - Vt.T @ MDlast)**2 + lambdaN * sum(eigenvalues_desc_order) + lambdaL*np.sum(abs(Dnew))

    out = {}
    out["optVal"]  = optVal
    out["count"]   = counterr
    out["Dlts1"]   = Dlts1
    out["Dlts2"]   = Dlts2
    out["Blast"]   = Bnew
    out["Clast"]   = Cnew
    out["Dlast"]   = Dnew
    out["B"]       = Dnew

    return out


final_dict = calculate_spinner_both(ytilde, SVDAx, lambdaN, lambdaL, W)



#np.unique(final_dict["Dlts2"])