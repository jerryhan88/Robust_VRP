import os.path as opath
import os
import csv, pickle
from functools import reduce
#
from mallTravelTime import sce_dpath
from problems import scenario_loader

exp_dpath = '_experiments'


def run(postfix):
    first_day, last_day = 1e400, -1e400
    t_uhij = []
    for fn in (os.listdir(sce_dpath)):
        if not fn.endswith('%s.pkl' % postfix): continue
        if 'opti' in fn: continue
        if 'mean' in fn: continue
        if 'pess' in fn: continue
        if 'rb' in fn: continue
        sn, _, _ = fn[:-len('.pkl')].split('-')
        day = int(sn[len('s_yyyymm'):])
        if day < first_day:
            first_day = day
        if day > last_day:
            last_day = day
        fpath = opath.join(sce_dpath, fn)
        scenario = scenario_loader(fpath)
        t_uhij.append(scenario['t_hij'])
    U = list(range(len(t_uhij)))
    #
    durStr = '%02d%02d' % (first_day, last_day)
    inputs_fpath = reduce(opath.join, [exp_dpath, 'input',
                            'input-%s-%s.pkl' % ('s_rb%s' % durStr, postfix)])
    with open(inputs_fpath, 'rb') as fp:
        inputs = pickle.load(fp)
    #
    sol_dpath = opath.join(exp_dpath, 'sol')
    solPKL = {}
    for strategy in ['mean', 'opti', 'pess']:
        solPKL_fpath = opath.join(sol_dpath, 'sol-s_%s%s-%s-obj1.pkl' % (strategy, durStr, postfix))
        with open(solPKL_fpath, 'rb') as fp:
            sol = pickle.load(fp)
        solPKL[strategy] = sol
    solPKL_fpath = opath.join(sol_dpath, 'sol-s_rb%s-%s.pkl' % (durStr, postfix))
    with open(solPKL_fpath, 'rb') as fp:
        sol = pickle.load(fp)
    rb_W1 = sol['W1']
    #
    V, D, n0, H, Ds, l_d, cT, p_d = [inputs.get(k) for k in ['V', 'D', 'n0', 'H', 'Ds', 'l_d', 'cT', 'p_sd']]
    ori_W1, actual_W1 = {}, {}
    for strategy, sol in solPKL.items():
        s_d, y_vd, x_hvdd, a_d, w_d, W1 = [sol.get(k) for k in ['s_d', 'y_vd', 'x_hvdd', 'a_d', 'w_d', 'W1']]
        ori_W1[strategy] = W1
        actual_W1[strategy] = -1e400
        for v in V:
            _route, tranTime = {}, {}
            for h in H:
                for d1 in Ds:
                    for d2 in Ds:
                        if x_hvdd[h, v, d1, d2] > 0.5:
                            _route[d1] = d2
                            tranTime[d1] = h
            route = [_route[n0]]
            while True:
                next_n = _route[route[-1]]
                if next_n == n0:
                    break
                route.append(next_n)
            for i in range(len(route) - 1):
                d0, d1 = route[i], route[i + 1]
                h0 = tranTime[d0]
                for u in U:
                    w1_d = cT * s_d[d1] - (cT * (s_d[d0] + p_d[d0]) + t_uhij[u][h0][l_d[d0]][l_d[d1]])
                    assert w1_d >= 0
                    if actual_W1[strategy] < w1_d:
                        actual_W1[strategy] = w1_d
    print('Robust: %.4f' % rb_W1)
    for strategy, W1 in ori_W1.items():
        print('\t %s: original %.4f, actual %.4f' % (strategy, W1, actual_W1[strategy]))


def summary():
    vrptw_sim_dpath = '_vrptw_simulation'
    sol_dpath = opath.join(vrptw_sim_dpath, 'sol')
    summary_fpath = opath.join(vrptw_sim_dpath, 'summary_vrptw_sim.csv')
    with open(summary_fpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        header = ['strategy', 'seedNum', 'objV', 'eliCpuTime', 'eliWallTime', 'fn']
        writer.writerow(header)

    for fn in os.listdir(sol_dpath):
        if not fn.endswith('.csv'):
            continue
        _, _, stg, seedNum, _, _ = fn[:-len('.csv')].split('-')
        with open(opath.join(sol_dpath, fn)) as r_csvfile:
            reader = csv.DictReader(r_csvfile)
            for row in reader:
                objV, eliCpuTime, eliWallTime = [row[cn] for cn in ['objV', 'eliCpuTime', 'eliWallTime']]
        with open(summary_fpath, 'a') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            writer.writerow([stg, seedNum, objV, eliCpuTime, eliWallTime, fn])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
FIGSIZE = (8, 4)

_fontsize = 14





def boxPlots():
    vrptw_sim_dpath = '_vrptw_simulation'
    summary_fpath = opath.join(vrptw_sim_dpath, 'summary_vrptw_sim.csv')
    df = pd.read_csv(summary_fpath)

    labels = ['Robust', 'Optimistic', 'Neutral', 'Pessimistic']
    labelConverter = {
        'Pessimistic': 'pess',
        'Neutral': 'mean',
        'Optimistic': 'opti',
        'Robust': 'rb'
    }

    data = []
    for stg in labels:
        stg_df = df[(df['strategy'] == labelConverter[stg])]
        objs = stg_df['objV']
        data.append(objs)
        print(stg, np.mean(objs), np.std(objs))



    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    # ax.set_xlabel('Parameter setting', fontsize=_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=_fontsize)
    medianprops = dict(linestyle='-', linewidth=1.0)
    boxprops = dict(linestyle='--', linewidth=1.0)
    plt.boxplot(data, boxprops=boxprops, medianprops=medianprops, labels=labels)

    img_ofpath = opath.join(vrptw_sim_dpath, 'boxplot_simRes.pdf')
    plt.savefig(img_ofpath, bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    # postfix = 'nd015-nv008'
    # run(postfix)

    summary()
    boxPlots()