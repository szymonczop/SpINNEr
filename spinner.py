import numpy as np

# stworzenie macierzy  AA czyli wielu macierzy w jednej.
n = 20
p = 5
y =np.array([4.7011,0.0157, 2.7057,-1.4226,5.5819,-3.7481, 10.0588,0.3313,-4.1411,5.1575,-6.2244, -3.1968,4.7178,
             0.6485,-5.5586,-5.9589,-2.5582,2.5288,-0.8932,-2.6688])

# macierz cech pacjetów
np.random.seed(2021)
X = np.random.randint(0, 20, size=(n,7))

lambdaN = 1.2
lambdaL = 0.9

# data = AA.copy()
# with open("best_file.txt", 'w') as outfile:
#     for data_slice in data:
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')

# np.random.seed(2021)
# AA = np.random.randint(0, 10, size=(n, p, p))
# for nr, matrix in enumerate(AA):
#     np.fill_diagonal(matrix, 0)
#     AA[nr, :, :] = (AA[nr, :, :] + AA[nr, :, :].T) / 2

AA= np.loadtxt('array_file.txt')
AA = AA.reshape(20,5,5)

AA.shape  # (20, 5, 5) (n, p, p)
# do tego momentu wszystkie pojedyńcze macierze w A są symetryczne i mają 0 na przekątnych


# chcecking symmetricity

for matrix in AA:
    if not np.allclose(matrix, matrix.T):
        print("One of A_i matrix is not symmetric")

    if np.diag(matrix).sum()>0:
        print("One of A_i has non zero value on diagonal")

# checking if X was provided
#To muszę dorobić

#Checking: dimmension check
if AA.shape[0] != len(y):
    print('The third dimension of 3-way tensor containing Ai`s should be the same as the length of y')

#Checking: dimmension check
if X.shape[0] != len(y):
    print('Number of rows in X and the length of y should coincide')

#Objects
nargin =2
p = AA.shape[1]
n = len(y)
if nargin == 5: ## NIE WIEM CO TA ZMIENNA NIBY ROBI
    if lambdaL > 0:
        W = np.ones((p,p)) - np.eye(p,p)
    else:
        W   = np.ones((p,p))




##########
########## get rid of X from the optimization problem
#########


H = np.eye(n) - X @ np.linalg.inv(X.T @ X) @ X.T
AAmatrix = AA.reshape(n,-1)
#AAmatrix.shape # 20x25  nxp^2
AAtilde = H @ AAmatrix
#AAtilde.shape # 20,25
AAtilde = AAtilde.reshape(n,p,p)
#AAtilde.shape 10 x 5 x 5
ytilde = H @ y

##############SVD
##############Convert the [p, p, n] array into a (p^2-p)/2-by-n matrix
##############
Avecs = AAtilde.reshape(n,-1)
#Avecs.shape 20x25
Avecs = Avecs.T # transponuje bo osoby w kolumnie osoby w wierszach cechy

upper_diagonal = np.triu(np.ones((p,p)),1).reshape(p**2,1) # to ma [(p-1)p]/2 jedynek
"""To jest bardzo sprytne, generalnie np.triu(np.ones((p,p)),1) dostaje macierz górnotrójkątną z 1 , 
tak naprawdę chce dostać tylko te kolumny uformowane z macierzy AA w których są te wartości z górnego trójkąta. Teraz 
 rozwijając je mam niejako jedynki w miejscach tych dobrych kolumn  i stosuje to do AvecsUP"""
idxs = [x[0] for x in upper_diagonal==1] # 10 wartości na którcyh mi zależy
#AvecsUP = 2 * Avecs[:,idxs] # Avecs to teraz macierz gdzie w wierszach są pacjęci a AvecsUP to tylko kolumny które
# są stworzone z wartości górnotrójkątnych macierzy A  pomnożone x2
#AvecsUP.shape # 10x10 czyli (p^2-p)/2-by-n czyli (5^2-5)/2-by-10) :)
AvecsUP = 2 * Avecs[idxs,:]
#AvecsUP.shape (10, 20)  (p^2-p)/2-by-n


# Economy-size SVD
U, Sdiag, Vt = np.linalg.svd(AvecsUP)  # U is (p^2-p)/2-by-n, S is n-by-n, V is n-by-n. # dostaje też inne wartośći niż założone
S = np.diag(Sdiag)
print(f"U shape{U.shape}, S shape {S.shape}, Vt shape {Vt.shape}") # te wymiary nie zgadzają mi sie z założeniami bbb
np.allclose(AvecsUP, U[:,:Sdiag.shape[0]] @ S @ Vt[:Sdiag.shape[0],:])

#########
######### SVD objects
#########
SVDAx = {}
SVDAx["U"] = U[:,:S.shape[0]] # czy tutaj zapisuje to co nie mnożymy przez 0 ?
middle_product = U[:,:S.shape[0]] @ S # złożenie pierwszych 2 macierzy
SVDAx["Sdiag"] = Sdiag
SVDAx["Vt"] =Vt[:middle_product.shape[0],:]
SVDAx["idxs"]  = idxs

## Cases
solverType = [x>0 for x in [lambdaN, lambdaL]]
solverType = solverType[0] + 2*solverType[1] + 1


## Solver

if solverType == 1:
    pass
elif solverType == 2:
    pass
elif solverType == 3:
    pass
elif solverType == 4:
    pass

## Estimates






