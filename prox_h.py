import numpy as np

def prox_H(B, C, delta1, delta2, Z1, Z2, lambda_N, WGTs):

    deltas = delta1 + delta2
    b_delta1 = B - Z1 / delta1
    b_delta2 = C - Z2 / delta2
    b_delta = (delta1 * b_delta1 + delta2 * b_delta2) / deltas
    d_new = np.sign(b_delta) * np.maximum(abs(b_delta) - lambda_N * WGTs / deltas, 0)
    return d_new

if __name__=="__main__":
    B = np.random.randint(0, 10, size=(5, 5))
    C = np.random.randint(0, 10, size=(5, 5))
    delta1 = 0.5
    delta2 = 0.7
    lambda_n = 2.5
    Z1 = np.random.randint(0, 5, size=(5, 5))
    Z2 = np.random.randint(0, 5, size=(5, 5))
    WGTs = np.ones((5, 5)) - np.eye(5)
    print(prox_H(B, C, delta1, delta2, Z1, Z2, lambda_n, WGTs))

