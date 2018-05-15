import os.path as opath
import multiprocessing
import time
import csv, pickle
from gurobipy import *
#
from problems import scenario_loader
#
NUM_CORES = multiprocessing.cpu_count()


def run(scenario, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    #
    problemName = scenario['problemName']
    n0, V, H, cT = [scenario.get(k) for k in ['n0', 'V', 'H', 'cT']]
    N, Ns, c_i = [scenario.get(k) for k in ['N', 'Ns', 'c_i']]
    D, Ds, l_d, Di = [scenario.get(k) for k in ['D', 'Ds', 'l_d', 'Di']]
    p_d, t_hij = [scenario.get(k) for k in ['p_d', 't_hij']]
    M1, M2 = [scenario.get(k) for k in ['M1', 'M2']]
    #
    s_d = scenario['s_d']
    #
    VM = Model('VM')
    y_vd = {(v, d): VM.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (v, d))
            for v in V for d in D}
    x_hvdd = {(h, v, d1, d2): VM.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d,%d]' % (h, v, d1, d2))
              for v in V for h in H for d1 in Ds for d2 in Ds}
    a_d, w_d = {}, {}
    for d in D:
        a_d[d] = VM.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % d)
        w_d[d] = VM.addVar(vtype=GRB.CONTINUOUS, name='w[%d]' % d)
    #
    W1 = VM.addVar(vtype=GRB.CONTINUOUS, name='W1')
    VM.update()
    #
    # Define constraints related to vehicle routing
    #
    for d in D:
        VM.addConstr(quicksum(y_vd[v, d] for v in V) == 1, name='d2v[%d]' % d)
    for v in V:
        for i in N:
            VM.addConstr(quicksum(y_vd[v, d] for d in Di[i]) <= 1, name='xSameLoc[%d,%d]' % (v, i))
    for v in V:
        VM.addConstr(quicksum(x_hvdd[h, v, n0, d] for h in H for d in Ds) == 1, name='DpoFlowO[%d]' % v)
        VM.addConstr(quicksum(x_hvdd[h, v, d, n0] for h in H for d in Ds) == 1, name='DpoFlowI[%d]' % v)
        for d1 in D:
            VM.addConstr(quicksum(x_hvdd[h, v, d1, d2] for h in H for d2 in Ds) == y_vd[v, d1],
                         name='OF_ASG[%d,%d]' % (v, d1))
            VM.addConstr(quicksum(x_hvdd[h, v, d2, d1] for h in H for d2 in Ds) == y_vd[v, d1],
                         name='IF_ASG[%d,%d]' % (v, d1))
    for d1 in D:
        VM.addConstr(a_d[d1] <= cT * s_d[d1], name='beforeST[%d]' % d1)
        for h in H:
            VM.addConstr(w_d[d1] <= 0 + M2 * (1 - quicksum(x_hvdd[h, v, n0, d1] for v in V)),
                         name='zeroWT[%d,%d]' % (h, d1))
            for d2 in D:
                VM.addConstr(cT * (s_d[d1] + p_d[d1]) + t_hij[h][l_d[d1]][l_d[d2]] \
                             <= a_d[d2] + M2 * (1 - quicksum(x_hvdd[h, v, d1, d2] for v in V)),
                             name='AT_LB2[%d,%d,%d]' % (h, d1, d2))
                VM.addConstr(a_d[d2] \
                             <= cT * h + t_hij[h][l_d[d1]][l_d[d2]] + M2 * (1 - quicksum(x_hvdd[h, v, d1, d2] for v in V)),
                             name='AT_UB2[%d,%d,%d]' % (h, d1, d2))
            VM.addConstr(h <= s_d[d1] + p_d[d1] + M1 * (1 - quicksum(x_hvdd[h, v, d1, d2] for v in V for d2 in D)),
                         name='inTS[%d,%d]' % (h, d1))
    for d in D:
        VM.addConstr(cT * s_d[d] - w_d[d] == a_d[d], name='calAT[%d]' % d)
    #
    # Objectives calculation
    #
    #  # OBJ 1
    for d in D:
        VM.addConstr(w_d[d] <= W1, name='W1[%d]' % d)
    #
    # Set objective
    #
    obj = LinExpr()
    obj += W1
    VM.setObjective(obj, GRB.MINIMIZE)
    #
    VM.setParam('Threads', NUM_CORES)
    if etc['logFile']:
        VM.setParam('LogFile', etc['logFile'])
    VM.optimize()
    #
    if VM.status == GRB.Status.INFEASIBLE:
        VM.write('%s.lp' % problemName)
        VM.computeIIS()
        VM.write('%s.ilp' % problemName)
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
            logContents += '\t ObjV: %.3f\n' % VM.objVal
            f.write(logContents)
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
            writer.writerow([VM.objVal, eliCpuTime, eliWallTime])
        #
        # Write pickle files recording scenario and the optimal solution
        #
        if not opath.exists(etc['inputFile']):
            with open(etc['inputFile'], 'wb') as fp:
                pickle.dump(scenario, fp)
        #
        _y_vd = {(v, d): y_vd[v, d].x for v in V for d in D}
        _x_hvdd = {(h, v, d1, d2): x_hvdd[h, v, d1, d2].x
                  for v in V for h in H for d1 in Ds for d2 in Ds}
        _a_d, _w_d = {}, {}
        for d in D:
            _a_d[d] = a_d[d].x
            _w_d[d] = w_d[d].x
        _W1 = W1.x
        sol = {
                'y_vd': _y_vd, 'x_hvdd': _x_hvdd,
                'a_d': _a_d, 'w_d': _w_d,
                #
                'W1': _W1}
        with open(etc['solFilePKL'], 'wb') as fp:
            pickle.dump(sol, fp)


def batch_run(target_dpath=None):
    if target_dpath == None:
        target_dpath = opath.join('_vrptw_scenarios', '_target')
    for fn in os.listdir(target_dpath):
        if not fn.endswith('.pkl'):
            continue
        target_exp_dpath = opath.join(target_dpath, '_experiments')
        if not opath.exists(target_exp_dpath):
            os.mkdir(target_exp_dpath)
        #
        prefix = fn[:-len('.pkl')]
        ifpath = opath.join(target_dpath, fn)
        input_dpath = opath.join(target_exp_dpath, 'input')
        sol_dpath = opath.join(target_exp_dpath, 'sol')
        log_dpath = opath.join(target_exp_dpath, 'log')
        for dpath in [input_dpath, sol_dpath, log_dpath]:
            if not opath.exists(dpath):
                os.mkdir(dpath)
        etc = {'inputFile': opath.join(input_dpath, 'input-%s.pkl' % prefix),
               'solFilePKL': opath.join(sol_dpath, 'sol-%s.pkl' % prefix),
               'solFileCSV': opath.join(sol_dpath, 'sol-%s.csv' % prefix),
               'solFileTXT': opath.join(sol_dpath, 'sol-%s.txt' % prefix),
               'logFile': opath.join(log_dpath, 'log-%s.txt' % prefix)}
        run(scenario_loader(ifpath), etc)


if __name__ == '__main__':
    batch_run()
