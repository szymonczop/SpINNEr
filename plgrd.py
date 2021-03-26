import numpy as np
import pandas as pd
import matplotlib.pyplot

D = np.random.randint(0,10,size=(5,5))
W2 = np.ones((5,5)) - np.eye(5)
delta = 0.5
lambda_n = 2.5

Ddelta = D + W2/delta
Ddelta = (Ddelta + Ddelta.T)/2
U, S, Vt = np.linalg.svd(Ddelta) # Ddelta = U*diag(S)*Vt
diagS = np.diag(S)
U @ diagS @ Vt
np.allclose(Ddelta, U @ diagS @ V)
Stsh =np.diag(np.sign(S) * np.maximum(np.abs(S) - lambda_n/delta,0)) # soft tresholoding of singular values
Cnew = U @ Stsh @ Vt

#############
#############
#############
B = np.random.randint(0,10,size=(5,5))
C = Cnew.copy()
delta1 = 0.5
delta2 = 0.7
Z1 = np.random.randint(0,5,size=(5,5))
Z2  = np.random.randint(0,5,size=(5,5))
WGTs = np.ones((5,5)) - np.eye(5)

deltas = delta1 + delta2
b_delta1 = B - Z1/delta1
b_delta2 = C - Z2/delta2
b_delta = (delta1 * b_delta1 + delta2 * b_delta2)/deltas
d_new = np.sign(b_delta) * np.maximum(abs(b_delta) - lambda_n*WGTs/deltas,0)


function Dnew = ProxH(B, C, delta1, delta2, W1, W2, lambda_L, WGTs)
    deltas     = delta1 + delta2;
    Bdelta1    = B - W1/delta1;
    Bdelta2    = C - W2/delta2;
    Bdelta     = (delta1*Bdelta1 + delta2*Bdelta2)/deltas;
    Dnew       = sign(Bdelta).*max(abs(Bdelta) - WGTs*lambda_L/deltas, 0);
end



from dotenv import load_dotenv


