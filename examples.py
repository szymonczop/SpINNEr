from spinner import calculate_spinner
from spinner_CV import calculate_CV
import numpy as np
import seaborn as sns
from scipy.linalg import block_diag
p = 40
n = 100
B1 = 2 * np.ones((5,5))
B2 = -2 * np.ones((5,5))
B3 = 2 * np.ones((4,4))
s_nods = B1.shape[0] + B2.shape[0] + B3.shape[0]
left_square = p - s_nods - 18 #
B = block_diag(np.zeros((5,5)), B1, np.zeros((6,6)), B2, np.zeros((7,7)),B3, np.zeros((left_square,left_square)) )
sns.heatmap(B, center=0, vmin=-2.5, vmax=2.5)

block_diag(np.zeros((5,5)), B1)

((outFinal["B"] - B)**2).sum()
