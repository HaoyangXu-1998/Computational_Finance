# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages 
# %%
def B_ARY(k, b):
    a = [0]
    if k > 0:
        j_max = int(np.floor(np.log(k) / np.log(b)))
        a = np.zeros(j_max + 1)
        q = b**j_max
        for i in range(a.shape[0]):
            a[i] = np.floor(k / q)
            k = k - q * a[i]
            q = q / b
    return np.array(a)
# %%
def NEXTB_ARY(a_in_old, b):
    a_in = a_in_old.copy()
    while a_in.shape[0] > 1 and a_in[0] == 0:
        a_in = a_in[1:]
    m = a_in.shape[0]
    carry = True
    a_out = np.zeros(m)
    for i in range(m-1, -1, -1):
        if carry:
            if a_in[i] == b - 1:
                a_out[i] = 0
            else:
                a_out[i] = a_in[i] + 1
                carry = False
        else:
            a_out[i] = a_in[i]
    if carry:
        a_out = np.concatenate(([1], a_out))
    return a_out
# %%
def getVanDerCorputPoints(n0, npts, b):
    def getOnePoint(a_in, b):
        x = 0
        q = 1/b
        m = len(a_in)
        for i in range(m):
            x += q * a_in[m-i-1]
            q /= b
        return x
    
    x = np.zeros(npts)
    a_in = B_ARY(n0, b)
    x[0] = getOnePoint(a_in, b)
    if npts != 1:
        for i in range(1, npts):
            a_in = NEXTB_ARY(a_in, b)
            x[i] = getOnePoint(a_in, b)
    return x
    
# %%
def FAUREMAT(r, i):
    C = np.diag(np.ones(r))
    if i > 0:
        C[0, :] = i**np.arange(0, r)
        for n in range(2, r):
            for m in range(1, n-2):
                C[m, n] = C[m-1, n-1] + i * C[m, n-1]
    return C

# %%
def FAUREPTS(n0, npts, d, b):
    n_max = n0 + npts - 1
    r_max = 1 + int(np.floor(np.log(n_max) / np.log(b)))
    b_pwrs = (1/b) ** np.arange(1, r_max+1)
    P = np.zeros((d, npts))
    a = np.zeros((r_max, npts))
    tmp = B_ARY(n0, b)
    if len(tmp) < r_max:
        tmp = np.concatenate(([0], tmp))
    a[:, 0] = tmp.copy()
    for i in range(1, npts):
        tmp = NEXTB_ARY(a[:, i-1], b)
        if len(tmp) < r_max:
            tmp = np.concatenate(([0], tmp))
        a[:, i] = tmp.copy()
    P[0, :] = np.matmul(b_pwrs, a)
    C = FAUREMAT(r_max, 1)
    for i in range(1, d):
        a = np.matmul(C, a) % b
        P[i, :] = np.matmul(b_pwrs, a)
    return P
# %%
def SOBOLMAT(c_vec, m_init, r):
    q = len(c_vec) - 1
    V = np.diag(np.ones(r))
    if q > 0:
        m_vec = np.concatenate((m_init, np.zeros(r - q))) 
        m_state = m_init
        for i in range(q, r):
            m_next = m_state[0]
            for j in range(1, q + 1):
                m_next ^= (2**j * int(c_vec[j]) * int(m_state[q-j]))
            m_vec[i] = m_next
            m_state = np.concatenate((m_state[1:], [m_next]))
        for j in range(r):
            m_bin = B_ARY(m_vec[j], 2)
            k = len(m_bin)
            for i in range(k):
                V[j-i, j] = m_bin[k-i-1]
    return V
# %%
def GRAYCODE2(n):
    tmp1 = B_ARY(n, 2)
    tmp2 = B_ARY(int(np.floor(n/2)), 2)
    if len(tmp2) < len(tmp1):
        tmp2 = np.concatenate((tmp2, np.zeros(len(tmp1) - len(tmp2))))
    return np.bitwise_xor(np.int32(tmp1), np.int32(tmp2))
# %%
def SOBOLPTS(n0, npts, d, p_vec, m_mat):
    n_max = n0 + npts - 1
    r_max = 1 + int(np.floor(np.log(n_max) / np.log(2)))
    r = 1
    P = np.zeros((npts, d))
    y = np.zeros((r_max, d))
    if n0 > 1:
        r = 1 + int(np.floor(np.log(n0 - 1) / np.log(2)))
    q_next = 2**r
    a = B_ARY(n0 - 1, 2)
    g = GRAYCODE2(n0 - 1)
    V = np.zeros((r_max, r_max, d))
    for i in range(d):
        q = int(np.floor(np.log(p_vec[i]) / np.log(2)))
        c_vec = B_ARY(p_vec[i], 2)
        # V[:, :, i] = SOBOLMAT(c_vec, m_mat[i, :], r_max)
        V[:, :, i] = SOBOLMAT(c_vec, m_mat[i], r_max)
    b_pwrs = (1/2)**np.arange(1, r_max + 1)
    for i in range(d):
        for m in range(r):
            for n in range(r):
                y[m, i] += (V[m, n, i] * g[r-n-1]) % 2
            # TODO: y[:, i] = (V[:, :, i] @ g[::-1]) % 2 might be ok?
    for k in range(n0, n_max + 1):
        if k == q_next:
            r += 1
            g = np.concatenate(([1], g))
            l = 0 # 0-based
            q_next *= 2
        else:
            for i in range(len(a) - 1, -1, -1):
                if a[i] == 0:
                    l = i
                    break
            g[l] = 1 - g[l]
        a = NEXTB_ARY(a, 2)
        for i in range(d):
            for m in range(r):
                # TODO: numpy accelaration
                y[m, i] += V[m, r-l-1, i]
                y[m, i] %= 2
            P[k-n0, i] = np.sum(b_pwrs * y[:, i])
    return P
# %%
def goodLatticePoints(npts, d, a=8363):
    n = 32749
    P = np.zeros((d, n))
    for i in range(npts+1):
        P[0, i] = i
        for j in range(1, d):
            P[j, i] = (a * P[j-1, i]) % n
    P /= n
    return P[:, 1:(npts+1)]
# %%
def L2_Star_Dis(P):
    d, npts = P.shape
    tmp1 = np.prod((1 - P**2), axis=1)
    tmp2 = np.zeros((npts, npts))
    for i in range(npts):
        for j in range(npts):
            tmp2[i, j] = np.prod(1 - np.max(P[:, [i, j]], axis=0))
    return (1/3)**d - (1/2)**(d-1) * np.mean(tmp1) + np.mean(tmp2)
# %%
T = 1
d = 8
r = 0.05
sigma = 0.3
S0 = 100
K = 100
# # %%
# s2 = (d + 1) * (2*d + 1)/(6*d**2) * sigma**2
# mu = (d + 1)/(2*d) * (r - sigma**2/2) + s2/2
# d1 = (np.log(S0/K) + (mu - s2/2) * T)/(np.sqrt(s2 * T))
# d2 = d1 - np.sqrt(s2 * T)
# C_BS = S0 * np.exp((mu - r) * T) * norm.cdf(d1) - K * norm.cdf(d2)
# %%
# Halton
def HaltonPricing(npts, T=1, d=8, r=0.05, sigma=0.3, S0=100, K=100):
    res = np.zeros(npts)
    dt = T/d
    t = np.arange(1, d + 1) * dt
    P = np.zeros((d, npts))
    b_list = [2, 3, 5, 7, 11, 13, 17, 19]
    for i in range(d):
        P[i, :] = getVanDerCorputPoints(1, npts, b_list[i])
    P = P.flatten().reshape((d, npts), order="F")
    
    for i in range(npts):
        p = P[:, i]
        z = norm.ppf(p)
        S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * np.sqrt(dt) * np.cumsum(z))
        res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
    return res
# %%
def FaurePricing(npts, T=1, d=8, r=0.05, sigma=0.3, S0=100, K=100):
    res = np.zeros(npts)
    dt = T/d
    t = np.arange(1, d + 1) * dt
    P = FAUREPTS(1, npts, d, 13)
    P = P.flatten().reshape((d, npts), order="F")
    for i in range(npts):
        p = P[:, i]
        z = norm.ppf(p)
        S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * np.sqrt(dt) * np.cumsum(z))
        res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
    return res
# # %%
# def SobolPricing(npts, T=1, d=8, r=0.05, sigma=0.3, S0=100, K=100):
#     res = np.zeros(npts)
#     dt = T/d
#     t = np.arange(1, d + 1) * dt
#     #TODO: fix bug here. Can numbers from the same order be used?
#     P = SOBOLPTS(1, npts, d, [3, 7, 11, 19, 37, 67, 131, 285], 
#     [[1, 1, 1, 1, 1], [1, 3, 5, 15, 17], [1, 1, 7, 11, 13], [1, 3, 7, 5, 7], [1, 1, 5, 3, 15], [1, 3, 1, 1, 9, 59, 25], [1, 1, 3, 7], [1, 3, 3, 9, 9]])
#     P = P.flatten().reshape((d, npts), order="F")
#     for i in range(npts):
#         p = P[:, i]
#         z = norm.ppf(p)
#         S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * np.sqrt(dt) * np.cumsum(z))
#         res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
#     return res
# %%
def MCPricing(npts, T=1, d=8, r=0.05, sigma=0.3, S0=100, K=100):
    res = np.zeros(npts)
    dt = T/d
    t = np.arange(1, d + 1) * dt
    for i in range(npts):
        z = norm.rvs(size=d)
        S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * np.sqrt(dt) * np.cumsum(z))
        res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
    return res
# %%
def getPrimeNumberList(d):
    p_list = [2]
    idx = 1
    n = 3
    while idx < d:
        flag = True
        for p in p_list:
            if n % p == 0:
                flag = False
                break
            if p**2 > n:
                break
        if flag:
            p_list.append(n)
            idx += 1
        n += 1
    return p_list

# %%
def generateBM_inner(n, Z, b0):
    # Brownian motion
    B = np.zeros(n+1)
    B[0] = b0
    h = n
    B[n] = B[0] + np.sqrt(h/n) * Z[0]
    idx = 1
    for k in range(1, int(np.floor(np.log(n) / np.log(2))) + 1):
        h = h // 2
        for j in range(1, int(2**(k-1)) + 1):
            B[(2*j-1)*h] = (B[2*(j-1)*h] + B[2*j*h]) / 2 + np.sqrt(h/2/n) * Z[idx]
            idx += 1
    return B
def generateBM(d, Z, b0_0):
    # Z: (d,)
    res = []
    while d != 0:
        i = 0
        while 2**i <= d:
            i += 1
        if not res:
            res.append(generateBM_inner(2**(i-1), Z[ :int(2**(i-1))], b0_0))
        else:
            res.append(generateBM_inner(2**(i-1), Z[ :int(2**(i-1))], res[-1][-1])[1:])
        d -= 2**(i-1)
        Z = Z[int(2**(i-1)): ]
    return np.concatenate(res)
# %%
def QMC(weights, method, sequence, npts, d=64, T=1, r=0.05, sigma=0.3, S0=100, K=100):
    res = np.zeros(npts)
    dt = T/d
    t = np.arange(1, d + 1) * dt
    if sequence == "Faure":
        b_list = getPrimeNumberList(d+1)
        P = FAUREPTS(b_list[-1]**4 - 1, npts*2, d+1, b_list[-1])[1:, npts:]
        # P = P.flatten(order="C").reshape((d, npts), order="F")
        Z = norm.ppf(P)
    elif sequence == "Halton":
        P = np.zeros((d, npts))
        b_list = getPrimeNumberList(d)
        for i in range(d):
            P[i, :] = getVanDerCorputPoints(1, npts, b_list[i])
        P = P.flatten(order="C").reshape((d, npts), order="F")
        Z = norm.ppf(P)
    elif sequence == "Korobov":
        P = goodLatticePoints(npts, d)
        Z = norm.ppf(P)
    elif sequence == "Standard":
        Z = norm.rvs(size=(d, npts))

    if method == "BB":
        for i in range(npts):
            B = generateBM(d, Z[:, i], 0)
            S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * B[1:])
            res[i] = np.exp(-r * T) * max(np.average(S, weights=weights) - K, 0)

    elif method == "PCA":
        C = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                C[i, j] = min(t[i], t[j])
        w, V = np.linalg.eig(C)
        A = np.matmul(V, np.diag(np.sqrt(w)))
        B = np.zeros((d, npts))
        for i in range(npts):
            B[:, i] = np.matmul(A, Z[:, i])
            S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * B[:, i])
            res[i] = np.exp(-r * T) * max(np.average(S, weights=weights) - K, 0)
    elif method == "Standard":
        for i in range(npts):
            S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * np.sqrt(dt) * np.cumsum(Z[:, i]))
            res[i] = np.exp(-r * T) * max(np.average(S, weights=weights) - K, 0)
    return res

# %%
# n_Faure_d=64
d = 64
w1 = np.ones(d) / d
npts_list = np.arange(500, 20500, 500)
res_faure = np.zeros((3, len(npts_list)))
std_faure = np.zeros((3, len(npts_list)))
for i, npts in enumerate(npts_list):
    res = QMC(w1, "BB", "Faure", npts, d)
    res_faure[0, i] = res.mean()
    std_faure[0, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "PCA", "Faure", npts, d)
    res_faure[1, i] = res.mean()
    std_faure[1, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "Standard", "Faure", npts, d)
    res_faure[2, i] = res.mean()
    std_faure[2, i] = np.sqrt(res.var() / npts)
# %%
p1 = PdfPages("./pre/figure/n_Faure_d=64.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(npts_list, res_faure[0, :], label="Faure+BB", linewidth=2)
plt.fill_between(npts_list, res_faure[0, :]-std_faure[0, :], res_faure[0, :]+std_faure[0, :], alpha=0.5)
plt.plot(npts_list, res_faure[1, :], label="Faure+PCA", linewidth=2)
plt.fill_between(npts_list, res_faure[1, :]-std_faure[1, :], res_faure[1, :]+std_faure[1, :], alpha=0.5)
plt.plot(npts_list, res_faure[2, :], label="Faure+Standard", linewidth=2)
plt.fill_between(npts_list, res_faure[2, :]-std_faure[2, :], res_faure[2, :]+std_faure[2, :], alpha=0.5)
plt.title("Faure Method with npts from 500 to 20000", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("npts", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()
# %%
# d_Faure_n=5000
npts = 5000
d_list = np.arange(8, 256+16, 16)
res_faure = np.zeros((3, len(d_list)))
std_faure = np.zeros((3, len(d_list)))
for i, d in enumerate(d_list):
    w1 = np.ones(d) / d
    res = QMC(w1, "BB", "Faure", npts, d)
    res_faure[0, i] = res.mean()
    std_faure[0, i] = res.std()
    res = QMC(w1, "PCA", "Faure", npts, d)
    res_faure[1, i] = res.mean()
    std_faure[1, i] = res.std()
    res = QMC(w1, "Standard", "Faure", npts, d)
    res_faure[2, i] = res.mean()
    std_faure[2, i] = res.std()
    print(w1)
std_faure /= np.sqrt(npts)
# %%
p1 = PdfPages("./pre/figure/d_Faure_n=5000.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(d_list, res_faure[0, :], label="Faure+BB", linewidth=2)
plt.fill_between(d_list, res_faure[0, :]-std_faure[0, :], res_faure[0, :]+std_faure[0, :], alpha=0.5)
plt.plot(d_list, res_faure[1, :], label="Faure+PCA", linewidth=2)
plt.fill_between(d_list, res_faure[1, :]-std_faure[1, :], res_faure[1, :]+std_faure[1, :], alpha=0.5)
plt.plot(d_list, res_faure[2, :], label="Faure+Standard", linewidth=2)
plt.fill_between(d_list, res_faure[2, :]-std_faure[2, :], res_faure[2, :]+std_faure[2, :], alpha=0.5)
plt.title("Faure Method with d from 8 to 256", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("d", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()
# %%
# n_Halton_d=64
d = 64
w1 = np.ones(d) / d
npts_list = np.arange(500, 20500, 500)
res_Halton = np.zeros((3, len(npts_list)))
std_Halton = np.zeros((3, len(npts_list)))
for i, npts in enumerate(npts_list):
    res = QMC(w1, "BB", "Halton", npts, d)
    res_Halton[0, i] = res.mean()
    std_Halton[0, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "PCA", "Halton", npts, d)
    res_Halton[1, i] = res.mean()
    std_Halton[1, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "Standard", "Halton", npts, d)
    res_Halton[2, i] = res.mean()
    std_Halton[2, i] = np.sqrt(res.var() / npts)
# %%
p1 = PdfPages("./pre/figure/n_Halton_d=64.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(npts_list, res_Halton[0, :], label="Halton+BB", linewidth=2)
plt.fill_between(npts_list, res_Halton[0, :]-std_Halton[0, :], res_Halton[0, :]+std_Halton[0, :], alpha=0.5)
plt.plot(npts_list, res_Halton[1, :], label="Halton+PCA", linewidth=2)
plt.fill_between(npts_list, res_Halton[1, :]-std_Halton[1, :], res_Halton[1, :]+std_Halton[1, :], alpha=0.5)
plt.plot(npts_list, res_Halton[2, :], label="Halton+Standard", linewidth=2)
plt.fill_between(npts_list, res_Halton[2, :]-std_Halton[2, :], res_Halton[2, :]+std_Halton[2, :], alpha=0.5)
plt.title("Halton Method with npts from 500 to 20000", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("npts", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()

# %%
# d_Halton_n=10000
npts = 10000
d_list = np.arange(8, 256+16, 16)
res_Halton = np.zeros((3, len(d_list)))
std_Halton = np.zeros((3, len(d_list)))
for i, d in enumerate(d_list):
    w1 = np.ones(d) / d
    res = QMC(w1, "BB", "Halton", npts, d)
    res_Halton[0, i] = res.mean()
    std_Halton[0, i] = res.std()
    res = QMC(w1, "PCA", "Halton", npts, d)
    res_Halton[1, i] = res.mean()
    std_Halton[1, i] = res.std()
    res = QMC(w1, "Standard", "Halton", npts, d)
    res_Halton[2, i] = res.mean()
    std_Halton[2, i] = res.std()
std_Halton /= np.sqrt(npts)
# %%
p1 = PdfPages("./pre/figure/d_Halton_n=10000.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(d_list, res_Halton[0, :], label="Halton+BB", linewidth=2)
plt.fill_between(d_list, res_Halton[0, :]-std_Halton[0, :], res_Halton[0, :]+std_Halton[0, :], alpha=0.5)
plt.plot(d_list, res_Halton[1, :], label="Halton+PCA", linewidth=2)
plt.fill_between(d_list, res_Halton[1, :]-std_Halton[1, :], res_Halton[1, :]+std_Halton[1, :], alpha=0.5)
plt.plot(d_list, res_Halton[2, :], label="Halton+Standard", linewidth=2)
plt.fill_between(d_list, res_Halton[2, :]-std_Halton[2, :], res_Halton[2, :]+std_Halton[2, :], alpha=0.5)
plt.title("Halton Method with d from 8 to 256", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("d", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()
# %%
# n_Standard_d=64
d = 64
w1 = np.ones(d) / d
npts_list = np.arange(500, 20500, 500)
res_Standard = np.zeros((3, len(npts_list)))
std_Standard = np.zeros((3, len(npts_list)))
for i, npts in enumerate(npts_list):
    res = QMC(w1, "BB", "Standard", npts, d)
    res_Standard[0, i] = res.mean()
    std_Standard[0, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "PCA", "Standard", npts, d)
    res_Standard[1, i] = res.mean()
    std_Standard[1, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "Standard", "Standard", npts, d)
    res_Standard[2, i] = res.mean()
    std_Standard[2, i] = np.sqrt(res.var() / npts)
# %%
p1 = PdfPages("./pre/figure/n_Standard_d=64.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(npts_list, res_Standard[0, :], label="Standard+BB", linewidth=2)
plt.fill_between(npts_list, res_Standard[0, :]-std_Standard[0, :], res_Standard[0, :]+std_Standard[0, :], alpha=0.5)
plt.plot(npts_list, res_Standard[1, :], label="Standard+PCA", linewidth=2)
plt.fill_between(npts_list, res_Standard[1, :]-std_Standard[1, :], res_Standard[1, :]+std_Standard[1, :], alpha=0.5)
plt.plot(npts_list, res_Standard[2, :], label="Standard+Standard", linewidth=2)
plt.fill_between(npts_list, res_Standard[2, :]-std_Standard[2, :], res_Standard[2, :]+std_Standard[2, :], alpha=0.5)
plt.title("Standard Method with npts from 500 to 20000", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("npts", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()

# %%
# %%
# d_Standard_n=10000
npts = 10000
d_list = np.arange(8, 256+16, 16)
res_Standard = np.zeros((3, len(d_list)))
std_Standard = np.zeros((3, len(d_list)))
for i, d in enumerate(d_list):
    w1 = np.ones(d) / d
    res = QMC(w1, "BB", "Standard", npts, d)
    res_Standard[0, i] = res.mean()
    std_Standard[0, i] = res.std()
    res = QMC(w1, "PCA", "Standard", npts, d)
    res_Standard[1, i] = res.mean()
    std_Standard[1, i] = res.std()
    res = QMC(w1, "Standard", "Standard", npts, d)
    res_Standard[2, i] = res.mean()
    std_Standard[2, i] = res.std()
std_Standard /= np.sqrt(npts)
# %%
p1 = PdfPages("./pre/figure/d_Standard_n=10000.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(d_list, res_Standard[0, :], label="Standard+BB", linewidth=2)
plt.fill_between(d_list, res_Standard[0, :]-std_Standard[0, :], res_Standard[0, :]+std_Standard[0, :], alpha=0.5)
plt.plot(d_list, res_Standard[1, :], label="Standard+PCA", linewidth=2)
plt.fill_between(d_list, res_Standard[1, :]-std_Standard[1, :], res_Standard[1, :]+std_Standard[1, :], alpha=0.5)
plt.plot(d_list, res_Standard[2, :], label="Standard+Standard", linewidth=2)
plt.fill_between(d_list, res_Standard[2, :]-std_Standard[2, :], res_Standard[2, :]+std_Standard[2, :], alpha=0.5)
plt.title("Standard Method with d from 8 to 256", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("d", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()
# %%
# n_Korobov_d=64
# TODO: fix bug here
d = 64
w1 = np.ones(d) / d
npts_list = np.arange(500, 20500, 500)
res_Korobov = np.zeros((3, len(npts_list)))
std_Korobov = np.zeros((3, len(npts_list)))
for i, npts in enumerate(npts_list):
    res = QMC(w1, "BB", "Korobov", npts, d)
    res_Korobov[0, i] = res.mean()
    std_Korobov[0, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "PCA", "Korobov", npts, d)
    res_Korobov[1, i] = res.mean()
    std_Korobov[1, i] = np.sqrt(res.var() / npts)
    res = QMC(w1, "Korobov", "Korobov", npts, d)
    res_Korobov[2, i] = res.mean()
    std_Korobov[2, i] = np.sqrt(res.var() / npts)
# %%
p1 = PdfPages("./pre/figure/n_Korobov_d=64.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(npts_list, res_Korobov[0, :], label="Korobov+BB", linewidth=2)
plt.fill_between(npts_list, res_Korobov[0, :]-std_Korobov[0, :], res_Korobov[0, :]+std_Korobov[0, :], alpha=0.5)
plt.plot(npts_list, res_Korobov[1, :], label="Korobov+PCA", linewidth=2)
plt.fill_between(npts_list, res_Korobov[1, :]-std_Korobov[1, :], res_Korobov[1, :]+std_Korobov[1, :], alpha=0.5)
plt.plot(npts_list, res_Korobov[2, :], label="Korobov+Korobov", linewidth=2)
plt.fill_between(npts_list, res_Korobov[2, :]-std_Korobov[2, :], res_Korobov[2, :]+std_Korobov[2, :], alpha=0.5)
plt.title("Korobov Method with npts from 500 to 20000", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("npts", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()

# %%
# d_Korobov_n=10000
npts = 10000
d_list = np.arange(8, 256+16, 16)
res_Korobov = np.zeros((3, len(d_list)))
std_Korobov = np.zeros((3, len(d_list)))
for i, d in enumerate(d_list):
    w1 = np.ones(d) / d
    res = QMC(w1, "BB", "Korobov", npts, d)
    res_Korobov[0, i] = res.mean()
    std_Korobov[0, i] = res.std()
    res = QMC(w1, "PCA", "Korobov", npts, d)
    res_Korobov[1, i] = res.mean()
    std_Korobov[1, i] = res.std()
    res = QMC(w1, "Korobov", "Korobov", npts, d)
    res_Korobov[2, i] = res.mean()
    std_Korobov[2, i] = res.std()
std_Korobov /= np.sqrt(npts)
# %%
p1 = PdfPages("./pre/figure/d_Korobov_n=10000.pdf")
f1 = plt.figure(figsize=(12, 8))
plt.plot(d_list, res_Korobov[0, :], label="Korobov+BB", linewidth=2)
plt.fill_between(d_list, res_Korobov[0, :]-std_Korobov[0, :], res_Korobov[0, :]+std_Korobov[0, :], alpha=0.5)
plt.plot(d_list, res_Korobov[1, :], label="Korobov+PCA", linewidth=2)
plt.fill_between(d_list, res_Korobov[1, :]-std_Korobov[1, :], res_Korobov[1, :]+std_Korobov[1, :], alpha=0.5)
plt.plot(d_list, res_Korobov[2, :], label="Korobov+Korobov", linewidth=2)
plt.fill_between(d_list, res_Korobov[2, :]-std_Korobov[2, :], res_Korobov[2, :]+std_Korobov[2, :], alpha=0.5)
plt.title("Korobov Method with d from 8 to 256", fontsize=24)
plt.legend(fontsize=20)
plt.ylabel("Price", fontsize=20)
plt.xlabel("d", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
p1.savefig()
p1.close()