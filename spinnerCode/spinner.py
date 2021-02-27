import numpy as np


# stworzenie macierzy  AA czyli wielu macierzy w jednej.
n = 20
p = 5
np.random.seed(2021)
AA = np.random.randint(0, 10, size=(n, p, p))

for nr, matrix in enumerate(AA):
    np.fill_diagonal(matrix, 0)
    AA[nr, :, :] = (AA[nr, :, :] + AA[nr, :, :].T) / 2

AA.shape  # (20, 5, 5) (n, p, p)
# do tego momentu wszystkie pojedyńcze macierze w A są symetryczne i mają 0 na przekątnych

##########
########## get rid of X from the optimization problem
#########
y = np.random.rand(n)
X = np.random.randint(0, 20, size=(n,7)) # to jest macierz cech pacjętów

H = np.eye(n) - X @ np.linalg.inv(X.T @ X) @ X.T
AAmatrix = AA.reshape(n,-1)
#AAmatrix.shape # 10x25  nxp^2
AAtilde = H @ AAmatrix
#AAtilde.shape # 10,25
AAtilde = AAtilde.reshape(n,p,p)
#AAtilde.shape 10 x 5 x 5
ytilde = H @ y

##############SVD
##############Convert the [p, p, n] array into a (p^2-p)/2-by-n matrix
##############
Avecs = AAtilde.reshape(n,-1)
#Avecs.shape 20x25
upper_diagonal = np.triu(np.ones((p,p)),1).reshape(p**2,-1) # to ma [(p-1)p]/2
"""To jest bardzo sprytne, generalnie np.triu(np.ones((p,p)),1) dostaje macierz górnotrójkątną z 1 , 
tak naprawdę chce dostać tylko te kolumny uformowane z macierzy AA w których są te wartości z górnego trójkąta. Teraz 
 rozwijając je mam niejako jedynki w miejscach tych dobrych kolumn  i stosuje to do AvecsUP"""
idxs = [x[0] for x in upper_diagonal==1]
AvecsUP = 2 * Avecs[:,idxs] # Avecs to teraz macierz gdzie w wierszach są pacjęci a AvecsUP to tylko kolumny które
# są stworzone z wartości górnotrójkątnych macierzy A  pomnożone x2
#AvecsUP.shape # 10x10 czyli (p^2-p)/2-by-n czyli (5^2-5)/2-by-10) :)

# Economy-size SVD
U, S, Vt = np.linalg.svd(AvecsUP)  # U is (p^2-p)/2-by-n, S is n-by-n, V is n-by-n.
Sdiag = np.diag(S)
np.allclose(AvecsUP, U[:,:Sdiag.shape[0]] @ Sdiag @ Vt)

#########
######### SVD objects
#########

SVDAx = {}
SVDAx["U"] = U[:,:Sdiag.shape[0]] # czy tutaj zapisuje to co nie mnożymy przez 0 ?
SVDAx["Sdiag"] = Sdiag
SVDAx["Vt"] = Vt
SVDAx["idxs"]  = idxs







