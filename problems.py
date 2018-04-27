import pandas as pd


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


def s0(retType='dict'):
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
    if retType == 'dict':
        return {'problemName': problemName,
                'n0': n0, 'V': V, 'H': H, 'cT': cT,
                'N': N, 'Ns': Ns, 'c_i': c_i,
                'D': D, 'Ds': Ds, 'l_d': l_d, 'Di': Di,
                'p_d': p_d, 't_hij': t_hij,
                'M1': M1, 'M2': M2
                }
    else:
        assert retType == 'tuple'
        return problemName, n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2


def s1(retType='dict'):
    problemName = 's1'
    _, n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, _, M1, M2 = s0(retType='tuple')
    #
    uT, nN = 0.5, len(N)
    t_hij = gen_t_hij(H, Ns, nN, uT)
    #
    if retType == 'dict':
        return {'problemName': problemName,
                'n0': n0, 'V': V, 'H': H, 'cT': cT,
                'N': N, 'Ns': Ns, 'c_i': c_i,
                'D': D, 'Ds': Ds, 'l_d': l_d, 'Di': Di,
                'p_d': p_d, 't_hij': t_hij,
                'M1': M1, 'M2': M2
                }
    else:
        assert retType == 'tuple'
        return problemName, n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2


def s2(retType='dict'):
    problemName = 's2'
    _, n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, _, _, M1, M2 = s0(retType='tuple')
    #
    p_d = [2, 3, 2, 1, 1, 1, 2, 2, 2, 2]
    #
    uT, nN = 0.5, len(N)
    t_hij = gen_t_hij(H, Ns, nN, uT)
    #
    if retType == 'dict':
        return {'problemName': problemName,
                'n0': n0, 'V': V, 'H': H, 'cT': cT,
                'N': N, 'Ns': Ns, 'c_i': c_i,
                'D': D, 'Ds': Ds, 'l_d': l_d, 'Di': Di,
                'p_d': p_d, 't_hij': t_hij,
                'M1': M1, 'M2': M2
                }
    else:
        assert retType == 'tuple'
        return problemName, n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2

MIN60 = 60

def es0(retType = 'dict'):
    problemName = 'es0_avg'
    mn_mid = {mn: i for i, mn in enumerate(['313', 'IMM', 'Tampines Mall'])}
    targeting_hours = list(range(8, 12))
    #
    nV, nH, nN, nD = 4, len(targeting_hours) * 2, len(mn_mid), 12
    assert nN <= nD
    #
    n0 = nD
    V = list(range(nV))
    H = list(range(nH))
    cT = 30  # min.
    #
    N = list(range(nN))
    Ns = list(range(n0 + 1))
    c_i = [2, 2, 2]
    #
    D = list(range(nD))
    Ds = D + [n0]
    l_d = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    Di = [[d for d in D if l_d[d] == i] for i in N]
    #
    p_d = [1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2]
    t_hij = []
    #
    ifpath = '/Users/JerryHan88/Dropbox/Data/_mallTravelTime_googleMaps.csv'
    df = pd.read_csv(ifpath)
    df['timeslot'] = df.apply(lambda row: row['hour'] * 2 + int(row['minutes'] / cT), axis=1)
    df = df.groupby(['fromMall', 'toMall', 'hour', 'timeslot']).mean()['duration'].reset_index()    
    df = df[df['hour'].isin(targeting_hours)]
    minTS = df['timeslot'].min()
    _t_h_ij = {}
    for fm, tm, _, _ts, duration in df.values:
        _t_h_ij[_ts - minTS, mn_mid[fm], mn_mid[tm]] = duration / MIN60
    for h in H:
        t_ij = []
        for i in Ns:
            t_j = [0 for _ in Ns]
            if i < nN or i == len(Ns) - 1:
                for j in Ns:
                    if i == j: continue
                    if (h, i, j) not in _t_h_ij: continue
                    t_j[j] = _t_h_ij[h, i, j]
            t_ij.append(t_j)
        t_hij.append(t_ij)
    #
    M1 = len(H)
    M2 = cT * len(H)
    #
    if retType == 'dict':
        return {'problemName': problemName,
                'n0': n0, 'V': V, 'H': H, 'cT': cT,
                'N': N, 'Ns': Ns, 'c_i': c_i,
                'D': D, 'Ds': Ds, 'l_d': l_d, 'Di': Di,
                'p_d': p_d, 't_hij': t_hij,
                'M1': M1, 'M2': M2
                }
    else:
        assert retType == 'tuple'
        return problemName, n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2


def ms_ex0():
    problemName = 's0s1s2'
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
    k_i = list(map(range, [1, 1, 2, 2, 1]))
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