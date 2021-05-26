# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
# %%
with open('C:\\Users\\Haoyang XU\\Desktop\\Quant\\data\\Data_V.pkl', 'rb') as f_v:
    data_v = pickle.load(f_v)


# %%
def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, np.array):
            ret.extend(i)
        else:
            ret.append(i)
    return ret

# %%
data_v = np.concatenate(data_v)
# %%
import pickle
import pandas as pd
import numpy as np
from numba import njit

# %%
class MatDec:
    def __init__(self, lr=1e-6, reg=1e-6, n_epochs=100, n_factors=20):
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_factors = n_factors

    def fit(self, X, V, X_val=None, V_val=None):
        bi = [np.mean(each) for each in V]
        bu = np.zeros(4000000)
        pu = np.random.normal(0, 1, (4000000, self.n_factors))
        qi = np.random.normal(0, 1, (10000, self.n_factors))
        r_bar = sum([sum(each) for each in V])/sum([len(each) for each in V])
        for m in range(self.n_epochs):
            print(m)
            for i in range(10000):
                for j in range(len(V[i])):
                    val = r_bar + bu[X[i][j]] + bi[i] + np.sum(pu[X[i][j], :] * qi[i, :])
                    err = V[i][j] - val
                    bu[X[i][j]] = bu[X[i][j]] + self.lr * (err - self.reg * bu[X[i][j]])
                    bi[i] = bi[i] + self.lr * (err - self.reg * bi[i])
                    pu[X[i][j], :], qi[i, :] = pu[X[i][j], :] + self.lr * (err * qi[i, :] - self.reg * pu[X[i][j], :]), \
                                               qi[i, :] + self.lr * (err * pu[X[i][j], :] - self.reg * qi[i, :])
# %%
with open('C:\\Users\\Haoyang XU\\Desktop\\Quant\\data\\Data_V.pkl', 'rb') as f_v:
    V = pickle.load(f_v)

with open('C:\\Users\\Haoyang XU\\Desktop\\Quant\\data\\Data_X.pkl', 'rb') as f_x:
    X = pickle.load(f_x)

# %%
m = MatDec()
m.fit(X ,V)
# %%
def generateBM(n, b0):
    # Brownian motion
    B = np.zeros(n+1)
    B[0] = b0

    # generating matrix
    A = np.zeros((n+1, n+1))

    #initialization
    h = n
    Z = np.random.randn()
    B[n] = B[0] + np.sqrt(h) * Z
    A[n, 1] = np.sqrt(h)
    idx = 1
    M = int(np.log2(h))
    for k in range(1, M+1):
        h = h // 2
        for j in range(1, 2**(k-1) + 1):
            z = np.random.randn()
            B[(2*j-1)*h] = (B[2*(j-1)*h] + B[2*j*h]) / 2 + np.sqrt(h/2) * z
            idx += 1

            A[(2*j-1)*h, :] = (A[2*(j-1)*h, :] + A[2*j*h, :]) / 2
            A[(2*j-1)*h, idx] += np.sqrt(h/2)
    return B, A[1:, 1: ]
# %%
B, A = generateBM(256, 0)
# %%
p1 = PdfPages("C:\\Users\\Haoyang XU\\Desktop\\CF_HW1\\p5_1.pdf")
plt.figure(figsize=(12, 8), dpi = 600)
plt.plot(B)
plt.title("$n={}$".format(256))
p1.savefig()
p1.close()

# %%
# if n is not the power of 2
n = 150
res = []
while n != 0:
    i = 0
    while 2**i <= n:
        i += 1
    if not res:
        res.append(generateBM(2**(i-1), 0)[0])
    else:
        res.append(generateBM(2**(i-1), res[-1][-1])[0][1:])
    n -= 2**(i-1)
res = np.concatenate(res)
# %%
p2 = PdfPages("C:\\Users\\Haoyang XU\\Desktop\\CF_HW1\\p5_2.pdf")
plt.figure(figsize=(12, 8), dpi = 600)
plt.plot(res)
plt.title("$n={}$".format(150))
p2.savefig()
p2.close()
# %%
A = generateBM(8, 0)[1]
np.savetxt("C:\\Users\\Haoyang XU\\Desktop\\CF_HW1\\p4.csv", A, '%.3f', ',')
# %%
def getRatio_rw(n):
    return np.arange(1, n+1) * np.arange(2*n, n, -1) / (n * (n + 1))
def getRatio_bb(A):
    tmp = np.sum(A**2, axis=0)
    return np.cumsum(tmp) / np.sum(tmp)
def getRatio_PCA(n):
    a = (np.arange(1, n+1) * 2 - 1) / (2 * n + 1) * np.pi / 2
    lambda_ = 1 / (np.sin(a))**2
    return np.cumsum(lambda_) / np.sum(lambda_)
# %%
# n = 16
res_16 = np.zeros((3, 5))
res_16[0, :] = getRatio_rw(16)[:5]
res_16[1, :] = getRatio_bb(generateBM(16, 0)[1])[:5]
res_16[2, :] = getRatio_PCA(16)[:5]
np.savetxt("C:\\Users\\Haoyang XU\\Desktop\\CF_HW1\\p7_16.csv", res_16, '%.4f', ',')
# %%
# n = 64
res_64 = np.zeros((3, 5))
res_64[0, :] = getRatio_rw(64)[:5]
res_64[1, :] = getRatio_bb(generateBM(64, 0)[1])[:5]
res_64[2, :] = getRatio_PCA(64)[:5]
np.savetxt("C:\\Users\\Haoyang XU\\Desktop\\CF_HW1\\p7_64.csv", res_64, '%.4f', ',')
# %%
# n = 256
res_256 = np.zeros((3, 5))
res_256[0, :] = getRatio_rw(256)[:5]
res_256[1, :] = getRatio_bb(generateBM(256, 0)[1])[:5]
res_256[2, :] = getRatio_PCA(256)[:5]
np.savetxt("C:\\Users\\Haoyang XU\\Desktop\\CF_HW1\\p7_256.csv", res_256, '%.4f', ',')
# %%
