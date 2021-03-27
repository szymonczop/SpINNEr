# Proximity operator for function H, where
###########################################################################
###                                                                   ###
###              H(D) := lambdaL*|| vec(W o D) ||_1                  ###
###                                                                    ###
###########################################################################

import numpy as np


def calc_proxH_lasso(B, delta, W, lambdaL, WGTs):
    Bdela =  B - W/delta
    change_matrix = abs(Bdela) - WGTs*lambdaL/delta
    change_matrix[change_matrix<0] = 0 # to mi zastępuje funcke max(x,0) w matlabie
    Dnew = np.sign(Bdela) * change_matrix
    return Dnew

#calc_proxH_lasso(Bnew, delta, Wk, lambdaL, W)

