import numpy as np
import math
from spinner import calculate_spinner

n = 100
p = 40

y = np.loadtxt("matrices/y_big.txt")
n = len(y)

AA = np.loadtxt("matrices/AA_big.txt")
AA = AA.reshape(n, p, p)

np.random.seed(2021)
X = np.random.randint(0, 20, size=(n, 7))

W = np.ones((p, p)) - np.eye(p, p)

#### Add parameters

use_parallel = False
defaultW = np.ones((p, p)) - np.eye(p, p)
grid_lengthN = 15
grid_lengthL = 15
kfolds = 5
display_status = True

### CV options
initi_lambda = 1
zero_search_ratio = 100
max_lambd_acc = 1e-2

### CV indices
minElemsN = math.floor(n / kfolds)
remaining_elems_N = n - minElemsN * kfolds
one_to_K_folds = np.array(range(1, kfolds + 1))
groups_idxs = np.tile(one_to_K_folds, (minElemsN, 1))
### tutaj w tym matlabie jest jakieś dingo dango i nie podoba mi się to , raczej nie będe tego potrzebować
groups_idxs = groups_idxs.T.reshape((-1,))  # takie coś potem miałem w matlabie
np.random.seed(2021)
random_sample = np.random.choice(100, 100)
groups_idxs = groups_idxs[random_sample]

# FINDING the maximal lambda L
c_lambdaL = initi_lambda
vals_lamd_L = []
counterr1 = 1
stop = 0

# finding lambda_L for which matrix of zeros is obtained
while True:
    out = calculate_spinner(y, AA, 0, c_lambdaL, W=W, X=X)
    print("OTO MÓJ COUNTER", counterr1)
    if np.sqrt(sum(sum(W @ out["B"] ** 2))) < 1e-16:
        break
    vals_lamd_L.append(c_lambdaL)
    c_lambdaL = zero_search_ratio * c_lambdaL
    counterr1 += 1
# initial interval for maximal lambda L
if len(vals_lamd_L) == 1:
    lam_L1 = 0
    lam_L2 = vals_lamd_L[0]
else:
    lam_L1 = vals_lamd_L[-2]
    lam_L2 = vals_lamd_L[-1]

### finding narrow interval form amximal lambda L
stop = 0
counterr2 = 1
vals_lamd_L_max = []

while True:
    c_lam_L_max_new = (lam_L1 + lam_L2) / 2
    out_new_o = calculate_spinner(y, AA, 0, c_lam_L_max_new, W=W, X=X)
    print("OTO MÓJ COUNTER", counterr2)

    if np.linalg.norm(out_new_o["B"], "fro") < 1e-16:
        lam_L2 = c_lam_L_max_new
    else:
        lam_L1 = c_lam_L_max_new
    vals_lamd_L_max.append(lam_L2)
    counterr2 += counterr2
    if abs(lam_L2 - lam_L1) / lam_L2 < max_lambd_acc:
        break

###### FINDING MAXIMAL LAMBDA N

c_lambdaN = initi_lambda
vals_lambda_N = []
counterr1 = 1
stopp = 0

## finding lambda_N for which matrix of zeros is obtained
while True:
    out = calculate_spinner(y, AA, c_lambdaN, 0, W=W, X=X)
    print("OTO MÓJ COUNTER", counterr1)
    if np.linalg.norm(out["B"], "fro") < 1e-16:
        break
    vals_lambda_N.append(c_lambdaN)
    c_lambdaN = zero_search_ratio * c_lambdaN
    counterr1 += 1

## initial interval for maxiaml lambda N
if len(vals_lambda_N) == 1:
    lam_N1 = 0
    lam_N2 = vals_lambda_N[0]
else:
    lam_N1 = vals_lambda_N[-2]
    lam_N2 = vals_lambda_N[-1]

### finding narrow interval for maximal lambda N
counterr2 = 1
vals_lamd_N_max = []
while True:
    c_lam_L_max_new = (lam_N1 + lam_L2) / 2
    out_new_o = calculate_spinner(y, AA, c_lam_L_max_new, 0, W=W, X=X)
    print("OTO MÓJ COUNTER", counterr2)
    if np.linalg.norm(out_new_o["B"], "fro") < 1e-16:
        lam_N2 = c_lam_L_max_new
    else:
        lam_N1 = c_lam_L_max_new
    vals_lamd_N_max.append(lam_N2)
    counterr2 += 1
    if abs(lam_N2 - lam_N1) / lam_N2 < max_lambd_acc:
        break

### final lambda grids

k = 0.75
seqq = np.array([x for x in range(1, grid_lengthL)]) / (grid_lengthL - 1)
lambs_L_grid = np.insert(np.exp((seqq * np.log(lam_L2 + 1) ** (1 / k)) ** k) - 1, 0, 0)
lambs_N_grid = np.insert(np.exp((seqq * np.log(lam_N2 + 1) ** (1 / k)) ** k) - 1, 0, 0)


### CRoss- validation
logliks_CV = np.zeros((grid_lengthN,grid_lengthL))
if use_parallel:
    pass
#TODO: To jest do implementacji zeby było paralell ale nie wiem czy chce mi się to robić

### optimal lambdas
#TODO: tak naprawdę bez tego co jest powyżej to nie ma zbytnio sensu dlatego dobieram po prostu pierwsze wartości
# jesli uzupełniłbym to pierwsze to tutaj też można się pobawić

best_lambda_L = lambs_L_grid[0]
best_lambda_N = lambs_N_grid[0]

### Final estimate
outFinal = calculate_spinner(y,AA,best_lambda_N,best_lambda_L, W =W, X =X)