# Proximity operator for function G, where
# G(C) := lambda_N*||C||_*
# ||C||_* - nuclear norm of C
import numpy as np


def prox_G(D, Z2, delta, lambda_N):
    d_delta = D + Z2 / delta
    d_delta = (d_delta + d_delta.T) / 2  # czemu tutaj to robimy ? wiem że symetryczy ale czy coś nas to kosztuje ?
    U, S, Vt = np.linalg.svd(d_delta)  # Ddelta = U*diag(S)*Vt
    diagS = np.diag(S)
    if np.allclose(d_delta, U @ diagS @ Vt) != True:
        print("ProxG SVD not made properly")
    Stsh = np.diag(np.sign(S) * np.maximum(np.abs(S) - lambda_N / delta, 0))  # soft tresholoding of singular values
    c_new = U @ Stsh @ Vt
    return c_new

if __name__=="__main__":
    D = np.random.randint(0, 10, size=(5, 5))
    Z2 = np.ones((5, 5)) - np.eye(5)
    delta = 0.5
    lambda_n = 2.5
    c_new = prox_G(D, Z2, delta, lambda_n)
    print(c_new)