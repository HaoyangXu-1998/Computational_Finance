# %%
import numpy as np
from scipy.stats import norm
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
def NEXTB_ARY(a_in, b):
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
    a[:, 0] = B_ARY(n0, b)
    for i in range(1, npts):
        a[:, i] = NEXTB_ARY(a[:, i-1], b)
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
                m_next ^= 2**j * int(c_vec[j]) * int(m_state[q-j])
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
    return np.bitwise_xor(np.int8(tmp1), np.int8(tmp2))
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
        V[:, :, i] = SOBOLMAT(c_vec, m_mat[i, :], r_max)
    b_pwrs = (1/2)**np.arange(1, r_max + 1)
    for i in range(d):
        for m in range(r):
            for n in range(r):
                y[m, i] += (V[m, n, i] * g[r-n-1]) % 2
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
                # y[m, i] += (V[m, r-l-1, i] % 2)
                y[m, i] += V[m, r-l-1, i]
                y[m, i] %= 2
            P[k-n0, i] = np.sum(b_pwrs * y[:, i])
    return P
# %%
def goodLatticePoints(npts, d=8, a=393):
    n = 2039
    P = np.zeros((d, n))
    for i in range(n):
        P[0, i] = i
        for j in range(1, d):
            P[j, i] = (a * P[j-1, i]) % n
    P /= n
    return P[:, :npts]
# %%
def L2_Star_Dis(P):
    # P: npts * d
    P = P.T
    npts, d = P.shape
    tmp1 = np.prod((1 - P**2), axis=1)
    tmp2 = np.zeros((npts, npts))
    for i in range(npts):
        for j in range(npts):
            tmp2[i, j] = np.prod(1 - np.max(P[[i, j], :], axis=0))
    return (1/3)**d - (1/2)**(d-1) * np.mean(tmp1) + np.mean(tmp2)
# %%
T = 1
d = 8
r = 0.05
sigma = 0.3
S0 = 100
K = 100
# %%
s2 = (d + 1) * (2*d + 1)/(6*d**2) * sigma**2
mu = (d + 1)/(2*d) * (r - sigma**2/2) + s2/2
d1 = (np.log(S0/K) + (mu - s2/2) * T)/(np.sqrt(s2 * T))
d2 = d1 - np.sqrt(s2 * T)
C_BS = S0 * np.exp((mu - r) * T) * norm.cdf(d1) - K * norm.cdf(d2)
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
# %%
def SobolPricing(npts, T=1, d=5, r=0.05, sigma=0.3, S0=100, K=100):
    res = np.zeros(npts)
    dt = T/d
    t = np.arange(1, d + 1) * dt
    P = SOBOLPTS(1, npts, d, [3, 7, 11, 19, 37], np.array([[1, 1, 1, 1, 1], [1, 3, 5, 15, 17], [1, 1, 7, 11, 13], [1, 3, 7, 5, 7], [1, 1, 5, 3, 15]]))
    P = P.flatten().reshape((d, npts), order="F")
    for i in range(npts):
        p = P[:, i]
        z = norm.ppf(p)
        S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * np.sqrt(dt) * np.cumsum(z))
        res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
    return res
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
def QMC(method, sequence, npts, T=1, d=8, r=0.05, sigma=0.3, S0=100, K=100):
    res = np.zeros(npts)
    dt = T/d
    t = np.arange(1, d + 1) * dt
    if sequence == "Faure":
        P = FAUREPTS(1, npts, d, 13)
        P = P.flatten().reshape((d, npts), order="F")
    elif sequence == "Halton":
        P = np.zeros((d, npts))
        b_list = [2, 3, 5, 7, 11, 13, 17, 19]
        for i in range(d):
            P[i, :] = getVanDerCorputPoints(1, npts, b_list[i])
        P = P.flatten().reshape((d, npts), order="F")
    elif sequence == "Korobov":
        P = goodLatticePoints(npts)
    if method == "BB":
        for i in range(npts):
            h = d
            Z = norm.ppf(P[:, i])
            B = np.zeros(d+1)
            B[d] = B[0] + np.sqrt(h/d) * Z[0]
            idx = 1
            for k in range(1, 4):
                h = h // 2
                for j in range(1, int(2**(k-1)) + 1):
                    B[(2*j-1)*h] = (B[2*(j-1)*h] + B[2*j*h]) / 2 + np.sqrt(h/2/d) * Z[idx]
                    idx += 1
            S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * B[1:])
            res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
    elif method == "PCA":
        C = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                C[i, j] = t[min(i, j)]
        w, V = np.linalg.eigh(C)
        w = w[::-1]
        V = V[:, ::-1]
        A = np.matmul(V, np.diag(np.sqrt(w)))
        for i in range(npts):
            B = np.dot(A, norm.ppf(P[:, i]))
            S = S0 * np.exp((r - 0.5*sigma**2) * t + sigma * B)
            res[i] = np.exp(-r * T) * max(np.prod(S**(1/d)) - K, 0)
    return res


# %%
