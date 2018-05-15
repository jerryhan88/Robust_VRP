import os.path as opath
import os
import pandas as pd
import pickle
from random import seed, randint, uniform
#
from mallTravelTime import sce_dpath
from mallTravelTime import TARGET_MALLS, TARGET_HOURS
from mallTravelTime import N_TS_HOUR, MIN15
#
SEED_NUM = 0
MIN_ST, MAX_ST = 1, 3
CAPA2 = 2

CAR_TRUCK_SPEED_RATIO = 1.5


def gen_scenario():
    seed(SEED_NUM)
    target_dates = ['20180507', '20180508', '20180509', '20180510', '20180511']
    #
    def get_base_scenario():
        nV, nH, nN, nD = 8, len(TARGET_HOURS) * N_TS_HOUR, len(TARGET_MALLS), 15
        assert nN <= nD
        assert (nD / nN) < nV
        #
        n0 = nD
        V = list(range(nV))
        H = list(range(nH))
        # cT = MIN30  # min.
        cT = MIN15  # min.
        #
        N = list(range(nN))
        Ns = list(range(n0 + 1))
        c_i = [CAPA2 for _ in N]
        #
        D = list(range(nD))
        Ds = D + [n0]
        l_d = [i % nN for i in range(nD)]
        Di = [[d for d in D if l_d[d] == i] for i in N]
        #
        p_d = [randint(MIN_ST, MAX_ST) for _ in range(nD)]
        # p_d = [1 for _ in range(nD)]
        while (nH * nN * CAPA2) * 0.8 < sum(p_d):
            p_d = [randint(MIN_ST, MAX_ST) for _ in range(nD)]
        #
        M1 = len(H)
        M2 = cT * len(H)
        #
        return n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, M1, M2
    #
    def get_t_hij(df, mn_mid, H, N, Ns):
        minTS = df['timeslot'].min()
        _t_h_ij = {}
        for fm, tm, durM, _ts in df.values:
            _t_h_ij[_ts - minTS, mn_mid[fm], mn_mid[tm]] = durM
        t_hij = []
        for h in H:
            t_ij = []
            for i in Ns:
                t_j = [0 for _ in Ns]
                if i < len(N) or i == len(Ns) - 1:
                    for j in Ns:
                        if i == j: continue
                        if (h, i, j) not in _t_h_ij: continue
                        t_j[j] = _t_h_ij[h, i, j] * CAR_TRUCK_SPEED_RATIO
                t_ij.append(t_j)
            t_hij.append(t_ij)
        #
        return t_hij
    #
    n0, V, H, cT, N, Ns, c_i, D, Ds, l_d, Di, p_d, M1, M2 = get_base_scenario()
    mn_mid = {mn: i for i, mn in enumerate(TARGET_MALLS)}
    #
    all_df = None
    raw_dpath = opath.join(sce_dpath, 'raw')
    postfix = 'nd%03d-nv%03d' % (len(D), len(V))
    for _date in target_dates:
        df = pd.read_csv(opath.join(raw_dpath, 'mTT-%s.csv' % _date))
        if all_df is None:
            all_df = pd.read_csv(opath.join(raw_dpath, 'mTT-%s.csv' % _date))
        else:
            all_df = all_df.append(df)
        df = df[df['hour'].isin(TARGET_HOURS)]
        df = df.drop(['duration', 'year', 'month', 'day', 'dow', 'hour', 'minute', 'Date'], axis=1)
        t_hij = get_t_hij(df, mn_mid, H, N, Ns)
        #
        problemName = 's_%s' % _date
        ofpath = opath.join(sce_dpath, '%s-%s.pkl' % (problemName, postfix))
        scenario = {'problemName': problemName,
                'n0': n0, 'V': V, 'H': H, 'cT': cT,
                'N': N, 'Ns': Ns, 'c_i': c_i,
                'D': D, 'Ds': Ds, 'l_d': l_d, 'Di': Di,
                'p_d': p_d, 't_hij': t_hij,
                'M1': M1, 'M2': M2
                }
        with open(ofpath, 'wb') as fp:
            pickle.dump(scenario, fp)
    first_day, last_day = 1e400, -1e400
    for dt_str in set(all_df['Date']):
        day = int(dt_str.split('-')[2])
        if day < first_day:
            first_day = day
        if day > last_day:
            last_day = day
    pess_df = all_df.groupby(['fromMall', 'toMall', 'timeslot']).max()['durM'].reset_index()
    opti_df = all_df.groupby(['fromMall', 'toMall', 'timeslot']).min()['durM'].reset_index()
    mean_df = all_df.groupby(['fromMall', 'toMall', 'timeslot']).mean()['durM'].reset_index()
    for strategy, df in [('pess', pess_df), ('opti', opti_df), ('mean', mean_df)]:
        df = df[['fromMall', 'toMall', 'durM', 'timeslot']]
        t_hij = get_t_hij(df, mn_mid, H, N, Ns)
        #
        problemName = 's_%s%02d%02d' % (strategy, first_day, last_day)
        ofpath = opath.join(sce_dpath, '%s-%s.pkl' % (problemName, postfix))
        scenario = {'problemName': problemName,
                  'n0': n0, 'V': V, 'H': H, 'cT': cT,
                  'N': N, 'Ns': Ns, 'c_i': c_i,
                  'D': D, 'Ds': Ds, 'l_d': l_d, 'Di': Di,
                  'p_d': p_d, 't_hij': t_hij,
                  'M1': M1, 'M2': M2
                  }
        with open(ofpath, 'wb') as fp:
            pickle.dump(scenario, fp)


def scenario_loader(fpath):
    with open(fpath, 'rb') as fp:
        scenario = pickle.load(fp)
    return scenario


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


def gen_rbProblem(postfix):
    candi_scenarios = []
    for fn in (os.listdir(sce_dpath)):
        if not fn.endswith('%s.pkl' % postfix): continue
        if 'opti' in fn: continue
        if 'mean' in fn: continue
        if 'pess' in fn: continue
        if 'rb' in fn: continue
        fpath = opath.join(sce_dpath, fn)
        scenario = scenario_loader(fpath)
        candi_scenarios.append((fn, scenario))
    #
    rb_scenarios = {}
    sns, p_sd, t_uhij = [], [], []
    for fn, scenario in candi_scenarios:
        sn, _, _ = fn[:-len('.pkl')].split('-')
        sns.append(sn)
        if not rb_scenarios:
            for k in ['n0', 'V', 'H', 'cT',
                         'N', 'Ns', 'c_i',
                         'D', 'Ds', 'l_d', 'Di',
                         'M1', 'M2']:
                rb_scenarios[k] = scenario[k]
        p_d, t_hij = [scenario.get(k) for k in ['p_d', 't_hij']]
        if not p_sd:
            p_sd = p_d[:]
        else:
            for i in range(len(p_sd)):
                if p_sd[i] < p_d[i]:
                    p_sd[i] = p_d[i]
        t_uhij.append(t_hij)
    rb_scenarios['U'] = list(range(len(sns)))
    rb_scenarios['p_sd'] = p_sd
    rb_scenarios['t_uhij'] = t_uhij
    #
    first_day, last_day = 1e400, -1e400
    for sn in sns:
        day = int(sn[len('s_yyyymm'):])
        if day < first_day:
            first_day = day
        if day > last_day:
            last_day = day
    problemName = 's_rb%02d%02d' % (first_day, last_day)
    rb_scenarios['problemName'] = problemName
    #
    ofpath = opath.join(sce_dpath, '%s-%s.pkl' % (problemName, postfix))
    with open(ofpath, 'wb') as fp:
        pickle.dump(rb_scenarios, fp)


def gen_vrptwProblem(postfix, numScenarios=10):
    strategies = ['opti', 'pess']
    strategy_fpath = {}
    durStr = None
    for fn in (os.listdir(sce_dpath)):
        if not fn.endswith('%s.pkl' % postfix): continue
        for stg in strategies:
            if stg in fn:
                if durStr is None:
                    durStr = fn[len('s_xxxx'):len('s_xxxx') + 4]
                strategy_fpath[stg] = opath.join(sce_dpath, fn)
    #
    base_scenario = {}
    strategy_t_hij = {}
    for stg in strategy_fpath:
        scenario = scenario_loader(strategy_fpath[stg])
        if not base_scenario:
            for k in ['n0', 'V', 'H', 'cT',
                      'N', 'Ns', 'c_i',
                      'D', 'Ds', 'l_d', 'Di',
                      'M1', 'M2',
                      'p_d']:
                base_scenario[k] = scenario[k]
        strategy_t_hij[stg] = scenario['t_hij']
    #
    exp_dpath = '_experiments'
    sol_dpath = opath.join(exp_dpath, 'sol')
    solPKL = {}
    for stg in ['opti', 'pess', 'mean']:
        solPKL_fpath = opath.join(sol_dpath, 'sol-s_%s%s-%s-obj1.pkl' % (stg, durStr, postfix))
        with open(solPKL_fpath, 'rb') as fp:
            sol = pickle.load(fp)
        solPKL[stg] = sol
    solPKL_fpath = opath.join(sol_dpath, 'sol-s_rb%s-%s.pkl' % (durStr, postfix))
    with open(solPKL_fpath, 'rb') as fp:
        sol = pickle.load(fp)
    solPKL['rb'] = sol
    stg_s_d = {}
    for stg in solPKL:
        stg_s_d[stg] =solPKL[stg]['s_d']
    #
    H, N, Ns = [base_scenario.get(k) for k in ['H', 'N', 'Ns']]
    opti_t_hij, pess_t_hij = [strategy_t_hij.get(strategy) for strategy in strategies]

    # print(durStr)
    vrptw_dpath = '_vrptw_scenarios'
    if not opath.exists(vrptw_dpath):
        os.mkdir(vrptw_dpath)
    for seedNum in range(numScenarios):
        seed(seedNum)
        t_hij = []
        for h in H:
            t_ij = []
            for i in Ns:
                t_j = [0 for _ in Ns]
                for j in Ns:
                    t_j[j] = uniform(opti_t_hij[h][i][j] * 0.95, pess_t_hij[h][i][j] * 1.05)
                t_ij.append(t_j)
            t_hij.append(t_ij)
        for stg, s_d in stg_s_d.items():
            vrptw_scenario = {}
            for k in ['n0', 'V', 'H', 'cT',
                      'N', 'Ns', 'c_i',
                      'D', 'Ds', 'l_d', 'Di',
                      'M1', 'M2',
                      'p_d']:
                vrptw_scenario[k] = base_scenario[k]
            vrptw_scenario['t_hij'] = t_hij
            vrptw_scenario['s_d'] = s_d
            problemName = 's_vrptw%s-%s-%d' % (durStr, stg, seedNum)
            vrptw_scenario['problemName'] = problemName
            #
            ofpath = opath.join(vrptw_dpath, '%s-%s.pkl' % (problemName, postfix))
            with open(ofpath, 'wb') as fp:
                pickle.dump(vrptw_scenario, fp)


if __name__ == '__main__':
    postfix = 'nd015-nv008'
    # gen_scenario()

    # gen_rbProblem(postfix)
    gen_vrptwProblem(postfix, 100)
