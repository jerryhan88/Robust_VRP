import os.path as opath
import os
import multiprocessing
import time
import csv, pickle
from gurobipy import *
#
from problems import scenario_loader
#
NUM_CORES = multiprocessing.cpu_count()
OBJ1, OBJ2, OBJ3, OBJ4, OBJ5, OBJ6 = range(1, 7)


def set_dvsSchedule(modelName, subInputs):
    MM = Model(modelName)
    D, H = subInputs
    s_d, e_d, z_hd = {}, {}, {}
    for d in D:
        s_d[d] = MM.addVar(vtype=GRB.INTEGER, name='s[%d]' % d)
        e_d[d] = MM.addVar(vtype=GRB.INTEGER, name='e[%d]' % d)
    for h in H:
        for d in D:
            z_hd[h, d] = MM.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (h, d))
    #
    return MM, s_d, e_d, z_hd


def get_dvsScheduleVal(subInputs, dvs):
    D, l_d, H = subInputs
    s_d, e_d, z_hd = dvs
    #
    _s_d, _e_d, _z_hd = {}, {}, {}
    for d in D:
        _s_d[d] = s_d[d].x
        _e_d[d] = e_d[d].x
    for h in H:
        for d in D:
            _z_hd[h, d] = z_hd[h, d].x
    #
    return _s_d, _e_d, _z_hd


def set_ctsScheduleDM(MM, subInputs, dvsSchedule):
    #
    # Define deterministic constraints related to time slot scheduling
    #
    H, N, c_i, D, l_d, Di, M1 = subInputs
    s_d, e_d, z_hd = dvsSchedule
    for d in D:
        MM.addConstr(0 <= s_d[d], name='tw_BE1[%d]' % d)
        MM.addConstr(e_d[d] <= len(H), name='tw_BE2[%d]' % d)
    for h in H:
        for d in D:
            MM.addConstr(s_d[d] <= h + M1 * (1 - z_hd[h, d]), name='startTS[%d,%d]' % (h, d))
            MM.addConstr(h <= e_d[d] + M1 * (1 - z_hd[h, d]), name='endTS[%d,%d]' % (h, d))
    for h in H:
        for i in N:
            MM.addConstr(quicksum(z_hd[h, d] for d in Di[i]) <= c_i[i], name='nodeCap[%d,%d]' % (h, i))


def run(inputs, targetOBj, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    #
    problemName = inputs['problemName']
    n0, V, H, cT = [inputs.get(k) for k in ['n0', 'V', 'H', 'cT']]
    N, Ns, c_i = [inputs.get(k) for k in ['N', 'Ns', 'c_i']]
    D, Ds, l_d, Di = [inputs.get(k) for k in ['D', 'Ds', 'l_d', 'Di']]
    p_d, t_hij = [inputs.get(k) for k in ['p_d', 't_hij']]
    M1, M2 = [inputs.get(k) for k in ['M1', 'M2']]
    #
    subInputs = (D, H)
    BM, s_d, e_d, z_hd = set_dvsSchedule('BM', subInputs)
    #
    y_vd = {(v, d): BM.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (v, d))
            for v in V for d in D}
    x_hvdd = {(h, v, d1, d2): BM.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d,%d]' % (h, v, d1, d2))
              for v in V for h in H for d1 in Ds for d2 in Ds}
    a_d, w_d = {}, {}
    for d in D:
        a_d[d] = BM.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % d)
        w_d[d] = BM.addVar(vtype=GRB.CONTINUOUS, name='w[%d]' % d)
    o_v = {v: BM.addVar(vtype=GRB.CONTINUOUS, name='o[%d]' % v) for v in V}
    c_v = {v: BM.addVar(vtype=GRB.CONTINUOUS, name='c[%d]' % v) for v in V}
    #
    W1 = BM.addVar(vtype=GRB.CONTINUOUS, name='W1')
    W2 = BM.addVar(vtype=GRB.CONTINUOUS, name='W2')
    S = BM.addVar(vtype=GRB.CONTINUOUS, name='S')
    WS1 = BM.addVar(vtype=GRB.CONTINUOUS, name='WS1')
    WS2 = BM.addVar(vtype=GRB.CONTINUOUS, name='WS2')
    J = BM.addVar(vtype=GRB.CONTINUOUS, name='J')
    BM.update()
    #
    # Define constraints related to time slot scheduling
    #
    subInputs = (H, N, c_i, D, l_d, Di, M1)
    dvsSchedule = (s_d, e_d, z_hd)
    set_ctsScheduleDM(BM, subInputs, dvsSchedule)
    for d in D:
        BM.addConstr(quicksum(z_hd[h, d] for h in H) == p_d[d], name='processingT[%d]' % d)
        BM.addConstr(s_d[d] + (p_d[d] - 1) == e_d[d], name='seTS_proT[%d]' % d)
    #
    # Define constraints related to vehicle routing
    #
    for d in D:
        BM.addConstr(quicksum(y_vd[v, d] for v in V) == 1, name='d2v[%d]' % d)
    for v in V:
        for i in N:
            BM.addConstr(quicksum(y_vd[v, d] for d in Di[i]) <= 1, name='xSameLoc[%d,%d]' % (v, i))
    for v in V:
        BM.addConstr(quicksum(x_hvdd[h, v, n0, d] for h in H for d in Ds) == 1, name='DpoFlowO[%d]' % v)
        BM.addConstr(quicksum(x_hvdd[h, v, d, n0] for h in H for d in Ds) == 1, name='DpoFlowI[%d]' % v)
        for d1 in D:
            BM.addConstr(quicksum(x_hvdd[h, v, d1, d2] for h in H for d2 in Ds) == y_vd[v, d1],
                         name='OF_ASG[%d,%d]' % (v, d1))
            BM.addConstr(quicksum(x_hvdd[h, v, d2, d1] for h in H for d2 in Ds) == y_vd[v, d1],
                         name='IF_ASG[%d,%d]' % (v, d1))
    for d1 in D:
        BM.addConstr(a_d[d1] <= cT * s_d[d1], name='beforeST[%d]' % d1)
        for h in H:
            # BM.addConstr(0 <= a_d[d1] + M2 * (1 - quicksum(x_hvdd[h, v, n0, d1] for v in V)),
            #              name='AT_LB1[%d,%d]' % (h, d1))
            BM.addConstr(w_d[d1] <= 0 + M2 * (1 - quicksum(x_hvdd[h, v, n0, d1] for v in V)),
                         name='zeroWT[%d,%d]' % (h, d1))
            for d2 in D:
                BM.addConstr(cT * (s_d[d1] + p_d[d1]) + t_hij[h][l_d[d1]][l_d[d2]] \
                             <= a_d[d2] + M2 * (1 - quicksum(x_hvdd[h, v, d1, d2] for v in V)),
                             name='AT_LB2[%d,%d,%d]' % (h, d1, d2))
                BM.addConstr(a_d[d2] \
                             <= cT * h + t_hij[h][l_d[d1]][l_d[d2]] + M2 * (1 - quicksum(x_hvdd[h, v, d1, d2] for v in V)),
                             name='AT_UB2[%d,%d,%d]' % (h, d1, d2))
            BM.addConstr(h <= s_d[d1] + p_d[d1] + M1 * (1 - quicksum(x_hvdd[h, v, d1, d2] for v in V for d2 in D)),
                         name='inTS[%d,%d]' % (h, d1))
    for d in D:
        BM.addConstr(cT * s_d[d] - w_d[d] == a_d[d], name='calAT[%d]' % d)
    #
    for v in V:
        for d in D:
            BM.addConstr(o_v[v] <= a_d[d] + M2 * (1 - y_vd[v, d]), name='vsUB[%d,%d]' % (v, d))
            BM.addConstr(cT * (s_d[d] + p_d[d]) <= c_v[v] + M2 * (1 - y_vd[v, d]), name='veLB[%d,%d]' % (v, d))
    #
    # Objectives calculation
    #
    #  # OBJ 1
    for d in D:
        BM.addConstr(w_d[d] <= W1, name='W1[%d]' % d)
    #  # OBJ 2
    for v in V:
        BM.addConstr(c_v[v] - o_v[v] <= J, name='J[%d]' % v)
    #  # OBJ 3
    BM.addConstr(quicksum(w_d[d] for d in D) == W2, name='W2')
    #  # OBJ 4
    BM.addConstr(quicksum(t_hij[h][l_d[d1]][l_d[d2]] * x_hvdd[h, v, d1, d2]
                          for v in V for h in H for d1 in D for d2 in D) == S, name='S')
    #  # OBJ 5
    BM.addConstr(W1 + S == WS1, name='WS1')
    #  # OBJ 6
    BM.addConstr(W2 + S == WS2, name='WS2')
    #
    # Set objective
    #
    obj = LinExpr()
    if targetOBj == OBJ1:
        obj += W1
    elif targetOBj == OBJ2:
        obj += J
    elif targetOBj == OBJ3:
        obj += W2
    elif targetOBj == OBJ4:
        obj += S
    elif targetOBj == OBJ5:
        obj += WS1
    else:
        assert targetOBj == OBJ6
        obj += WS2
    BM.setObjective(obj, GRB.MINIMIZE)
    #
    BM.setParam('Threads', NUM_CORES)
    if etc['logFile']:
        BM.setParam('LogFile', etc['logFile'])
    BM.optimize()
    #
    if BM.status == GRB.Status.INFEASIBLE:
        BM.write('%s.lp' % problemName)
        BM.computeIIS()
        BM.write('%s.ilp' % problemName)
    #
    if etc:
        assert 'inputFile' in etc
        assert 'solFilePKL' in etc
        assert 'solFileCSV' in etc
        assert 'solFileTXT' in etc
        #
        # Write a text file saving the optimal solution
        #
        endCpuTime, endWallTime = time.clock(), time.time()
        eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
        with open(etc['solFileTXT'], 'w') as f:
            f.write('The optimal solution of problem %s\n' % problemName)
            logContents = 'Summary\n'
            logContents += '\t Cpu Time: %f\n' % eliCpuTime
            logContents += '\t Wall Time: %f\n' % eliWallTime
            logContents += '\t ObjV: %.3f\n' % BM.objVal
            f.write(logContents)
            f.write('\n')
            f.write('Time slot scheduling\n')
            for d in D:
                f.write('\t D%d: TS [%02d, %02d]; \t WT %.2f\n' % (d, s_d[d].x, e_d[d].x, w_d[d].x))
            f.write('\n')
            f.write('Vehicle routing\n')
            for v in V:
                demand = []
                for d in D:
                    if y_vd[v, d].x > 0.5:
                        demand.append(d)
                _route = {}
                for h in H:
                    for d1 in Ds:
                        for d2 in Ds:
                            if x_hvdd[h, v, d1, d2].x > 0.5:
                                _route[d1] = d2
                route = [n0, _route[n0]]
                while route[-1] != n0:
                    route.append(_route[route[-1]])
                f.write('\t V%d: %s (%s);\n' % (v, str(demand), '->'.join(map(str, route))))
                f.write('\t\t\t\t\t (%s)\n' % '-'.join(['%.2f' % a_d[d].x for d in route[1:-1]]))
        #
        # Write a csv file
        #
        with open(etc['solFileCSV'], 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['objV', 'eliCpuTime', 'eliWallTime']
            writer.writerow(header)
            writer.writerow([BM.objVal, eliCpuTime, eliWallTime])
        #
        # Write pickle files recording inputs and the optimal solution
        #
        if not opath.exists(etc['inputFile']):
            with open(etc['inputFile'], 'wb') as fp:
                pickle.dump(inputs, fp)
        #
        subInputs = [D, l_d, H]
        dvs = [s_d, e_d, z_hd]
        _s_d, _e_d, _z_hd = get_dvsScheduleVal(subInputs, dvs)
        _y_vd = {(v, d): y_vd[v, d].x for v in V for d in D}
        _x_hvdd = {(h, v, d1, d2): x_hvdd[h, v, d1, d2].x
                  for v in V for h in H for d1 in Ds for d2 in Ds}
        _a_d, _w_d = {}, {}
        for d in D:
            _a_d[d] = a_d[d].x
            _w_d[d] = w_d[d].x
        _o_v, _c_v = {}, {}
        for v in V:
            _o_v[v] = o_v[v].x
            _c_v[v] = c_v[v].x
        _W1, _J, _W2, _S, _WS1, _WS2 = [dv.x for dv in [W1, J, W2, S, WS1, WS2]]
        sol = {'s_d': _s_d, 'e_d': _e_d, 'z_hd': _z_hd,
                #
                'y_vd': _y_vd, 'x_hvdd': _x_hvdd,
                'a_d': _a_d, 'w_d': _w_d,
               '_o_v': _o_v,'_c_v': _c_v,
                #
                'W1': _W1, 'J': _J, 'W2': _W2, 'S': _S, 'WS1': _WS1, 'WS2': _WS2}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


def batch_run(target_dpath=None):
    if target_dpath == None:
        target_dpath = opath.join('_scenario', '_target')
    exp_dpath = '_experiments'
    if not opath.exists(exp_dpath):
        os.mkdir(exp_dpath)
    input_dpath = opath.join(exp_dpath, 'input')
    sol_dpath = opath.join(exp_dpath, 'sol')
    for dpath in [input_dpath, sol_dpath]:
        if not opath.exists(dpath):
            os.mkdir(dpath)
    #
    for fn in os.listdir(target_dpath):
        if not fn.endswith('.pkl'):
            continue
        target_exp_dpath = opath.join(target_dpath, '_experiments')
        os.mkdir(target_exp_dpath)
        #
        prefix = fn[:-len('.pkl')]
        ifpath = opath.join(target_dpath, fn)
        input_dpath = opath.join(target_exp_dpath, 'input')
        sol_dpath = opath.join(target_exp_dpath, 'sol')
        log_dpath = opath.join(target_exp_dpath, 'log')
        for dpath in [input_dpath, sol_dpath, log_dpath]:
            os.mkdir(dpath)
        etc = {'inputFile': opath.join(input_dpath, 'input-%s.pkl' % prefix)}
        for targetObj in [OBJ1, OBJ2, OBJ3, OBJ4, OBJ5, OBJ6]:
            if targetObj not in [OBJ1]: continue
            # if targetObj not in [OBJ1, OBJ2]: continue
            # if targetObj in [OBJ4, OBJ5, OBJ6]: continue
            # if targetObj in [OBJ1, OBJ2, OBJ3]: continue
            etc['solFilePKL'] = opath.join(sol_dpath, 'sol-%s-obj%d.pkl' % (prefix, targetObj))
            etc['solFileCSV'] = opath.join(sol_dpath, 'sol-%s-obj%d.csv' % (prefix, targetObj))
            etc['solFileTXT'] = opath.join(sol_dpath, 'sol-%s-obj%d.txt' % (prefix, targetObj))
            etc['logFile'] = opath.join(log_dpath, 'log-%s-obj%d.txt' % (prefix, targetObj))
            run(scenario_loader(ifpath), targetObj, etc)


if __name__ == '__main__':
    from problems import s0, s1, s2

    # run(s0(), writing_files=True)
    # run(s1(), writing_files=True)
    # run(s2(), writing_files=True)
    # run(es0(), writing_files=True)

    batch_run()
