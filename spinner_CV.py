import numpy as np
import math
from spinner import calculate_spinner
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# n = 100
# p = 40

# y = np.loadtxt("matrices/y_big.txt")
# n = len(y)
#
# AA = np.loadtxt("matrices/AA_big.txt")
# AA = AA.reshape(n, p, p)
#
# np.random.seed(2021)
# X = np.random.randint(0, 20, size=(n, 7))
#
# W = np.ones((p, p)) - np.eye(p, p)


#### Add parameters

def calculate_CV(y, AA, W=None, X=None, use_parallel=False):
    n, p, _ = AA.shape
    n = len(y)

    # use_parallel = False
    # defaultW = np.ones((p, p)) - np.eye(p, p)
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
    while stop==0:
        out = calculate_spinner(y, AA, 0, c_lambdaL, W=W, X=X)
        if np.sqrt(((W * out["B"])** 2).sum()) < 1e-16:
            stop = 1
        vals_lamd_L.append(c_lambdaL)
        c_lambdaL = zero_search_ratio * c_lambdaL
        counterr1 += 1
        print("OTO MÓJ COUNTER", counterr1)
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
        counterr2 += 1
        if abs(lam_L2 - lam_L1) / lam_L2 < max_lambd_acc:
            break

    ###### FINDING MAXIMAL LAMBDA N

    c_lambdaN = initi_lambda
    vals_lambda_N = []
    counterr1 = 1
    stopp = 0

    ## finding lambda_N for which matrix of zeros is obtained
    while stopp==0:
        out = calculate_spinner(y, AA, c_lambdaN, 0, W=W, X=X)
        if np.linalg.norm(out["B"], "fro") < 1e-16:
            stopp=1
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
        c_lam_L_max_new = (lam_N1 + lam_N2) / 2
        out_new_o = calculate_spinner(y, AA, c_lam_L_max_new, 0, W=W, X=X)
        print("OTO MÓJ COUNTER", counterr2)
        if np.linalg.norm(out_new_o["B"], "fro") < 1e-16:
            lam_N2 = c_lam_L_max_new
        else:
            lam_N1 = c_lam_L_max_new
        vals_lamd_N_max.append(lam_N2)
        counterr2 += 1
        if counterr2>300:
            break
        if abs(lam_N2 - lam_N1) / lam_N2 < max_lambd_acc:
            break



    ### final lambda grids

    k = 0.75
    seqq = np.array([x for x in range(1, grid_lengthL)]) / (grid_lengthL - 1)
    lambs_L_grid = np.insert(np.exp((seqq * np.log(lam_L2 + 1) ** (1 / k)) ** k) - 1, 0, 0.0001)
    lambs_N_grid = np.insert(np.exp((seqq * np.log(lam_N2 + 1) ** (1 / k)) ** k) - 1, 0, 0.0001)


######### DO TEGO MOMENTU JEST TAK SAMO
    #### Tutaj trzeba zaimplementować to co jest podane wcześniej
    ### CRoss- validation
    logliks_CV = np.zeros((grid_lengthN, grid_lengthL))
    use_parallel = False
    if use_parallel:
        pass
    # TODO: To jest do implementacji zeby było paralell ale nie wiem czy chce mi się to robić

    ### optimal lambdas
    # TODO: tak naprawdę bez tego co jest powyżej to nie ma zbytnio sensu dlatego dobieram po prostu pierwsze wartości
    # jesli uzupełniłbym to pierwsze to tutaj też można się pobawić
    row = 0
    for i in tqdm(lambs_N_grid):
        col = 0
        c_lambdaN = i
        for j in lambs_L_grid:
            c_lambdaL = j
            norm_res_CV = []
            for g in range(1,kfolds+1):
                test_indices = np.where(groups_idxs==g)
                trening_indices = np.where(groups_idxs!=g)
                AA_test = AA[test_indices, :, :][0] # bo to bez [0] tworzy (1, 20, 40, 40)
                AA_trening = AA[trening_indices ,:,:][0]

                if X is not  None:
                    X_trening = X[trening_indices,:][0]
                    X_test = X[test_indices,:][0]

                else:
                    X_trening = []
                    X_test = 0

                y_trening = y[trening_indices]
                y_test = y[test_indices]
                out_cv = calculate_spinner(y_trening, AA_trening, c_lambdaN, c_lambdaL, X = X_trening)
                #AA_test_p = np.transpose(AA_test, (1,0,2))
                AA_test_p = AA_test.reshape(AA_test.shape[0],AA_test.shape[1] * AA_test.shape[2])
                val_for_norm_res = 0.5 * np.linalg.norm(y_test - AA_test_p @out_cv["B"].reshape(-1) - X_test@out_cv["beta"] )**2
                norm_res_CV.append(val_for_norm_res)

            logliks_CV[row, col] = sum(norm_res_CV)/n
            col += 1

        row += 1


    best_idx_lambs = np.where(logliks_CV == np.amin(logliks_CV))
    lambda_N_best_idx = best_idx_lambs[0]
    lambda_L_best_idx = best_idx_lambs[1]

    best_lambda_N = lambs_L_grid[lambda_N_best_idx]
    best_lambda_L = lambs_N_grid[lambda_L_best_idx]

    ### Final estimate
    outFinal = calculate_spinner(y, AA, best_lambda_N, best_lambda_L, W=W, X=X)
    sns.heatmap(outFinal["B"], center=0, vmin=-2.5, vmax=2.5)

    return outFinal

if __name__ == "__main__":
    y = np.loadtxt("matrices/y_big.txt")
    n = len(y)
    p = 40
    AA = np.loadtxt("matrices/AA_big.txt")
    AA = AA.reshape(n, p, p)

    np.random.seed(2021)
    X = np.random.randint(0, 20, size=(n, 7))

    W = np.ones((p, p)) - np.eye(p, p)

    spinner_fit = calculate_CV(y, AA, W=W, X=X, use_parallel=False)

    sns.heatmap(spinner_fit["B"], center=0, vmin= -2.5, vmax = 2.5)
    plt.show()