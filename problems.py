
def gen_t_hij(H, Ns, nN, uT):
    t_hij = []
    for _ in H:
        t_ij = []
        for i in Ns:
            t_j = [0 for _ in Ns]
            if i < nN or i == len(Ns) - 1:
                for j in Ns:
                    if i == j: continue
                    t_j[j] = uT
            t_ij.append(t_j)
        t_hij.append(t_ij)
    return t_hij


def s0():
    problemName = 's0'
    nV, nH, nN, nD, uT = 3, 12, 5, 10, 1
    assert nN <= nD
    #
    n0 = nD
    V = list(range(nV))
    H = list(range(nH))
    cT = 1
    #
    N = list(range(nN))
    Ns = list(range(n0 + 1))
    c_i = [1, 1, 1, 2, 2]
    k_i = map(list, map(range, [1, 1, 2, 2, 1]))
    T_i = [
            [(0, 5)],
            [(5, 11)],
            [(0, 2), (5, 8)],
            [(2, 5), (9, 11)],
            [(3, 10)],
          ]
    #
    D = list(range(nD))
    Ds = D + [n0]
    l_d = [0, 1, 2, 3, 4, 0, 3, 3, 4, 4]
    Di = [[d for d in D if l_d[d] == i] for i in N]
    #
    p_d = [1, 1, 2, 2, 1, 2, 2, 3, 3, 4]
    t_hij = gen_t_hij(H, Ns, nN, uT)
    #
    M1 = len(H)
    M2 = cT * len(H)
    #
    return problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2


def s1():
    problemName = 's1'
    _, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, p_d, _, M1, M2 = s0()
    #
    uT, nN = 0.5, len(N)
    t_hij = gen_t_hij(H, Ns, nN, uT)
    #
    return problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2


def s2():
    problemName = 's2'
    _, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, _, _, M1, M2 = s0()
    #
    p_d = [2, 3, 2, 1, 1, 1, 2, 2, 2, 2]
    #
    uT, nN = 0.5, len(N)
    t_hij = gen_t_hij(H, Ns, nN, uT)
    #
    return problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2


def ms_ex0():
    problemName = 's1s2s3'
    nV, nH, nN, nD, nU = 3, 12, 5, 10, 3
    assert nN <= nD
    #
    n0 = nD
    V = list(range(nV))
    H = list(range(nH))
    cT = 1
    #
    N = list(range(nN))
    Ns = list(range(n0 + 1))
    c_i = [1, 1, 1, 2, 2]
    k_i = map(list, map(range, [1, 1, 2, 2, 1]))
    T_i = [
            [(0, 5)],
            [(5, 11)],
            [(0, 2), (5, 8)],
            [(2, 5), (9, 11)],
            [(3, 10)],
          ]
    #
    D = list(range(nD))
    Ds = D + [n0]
    l_d = [0, 1, 2, 3, 4, 0, 3, 3, 4, 4]
    Di = [[d for d in D if l_d[d] == i] for i in N]
    #
    U = list(range(nU))
    p_ud = [[1, 1, 2, 2, 1, 2, 2, 3, 3, 4],
            [1, 1, 2, 2, 1, 2, 2, 3, 3, 4],
            [2, 3, 2, 1, 1, 1, 2, 2, 2, 2]]
    t_uhij = []
    for u in U:
        uT = 1.0 if u == 0 else 0.5
        t_hij = gen_t_hij(H, Ns, nN, uT)
        t_uhij.append(t_hij)
    #
    M1 = len(H)
    M2 = cT * len(H)
    #
    return problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, U, p_ud, t_uhij, M1, M2


if __name__ == '__main__':
    s0()