import os.path as opath
import multiprocessing
import time
import csv, pickle
from gurobipy import *
#
from problems import scenario_loader
from basicModel import set_dvsSchedule, set_ctsScheduleDM, get_dvsScheduleVal


NUM_CORES = multiprocessing.cpu_count()


def run(scenario, etc=None):
    startCpuTime, startWallTime = time.clock(), time.time()
    #
    problemName = scenario['problemName']
    n0, V, H, cT = [scenario.get(k) for k in ['n0', 'V', 'H', 'cT']]
    N, Ns, c_i = [scenario.get(k) for k in ['N', 'Ns', 'c_i']]
    D, Ds, l_d, Di = [scenario.get(k) for k in ['D', 'Ds', 'l_d', 'Di']]
    U, p_sd, t_uhij = [scenario.get(k) for k in ['U', 'p_sd', 't_uhij']]
    M1, M2 = [scenario.get(k) for k in ['M1', 'M2']]
    #
    subInputs = (D, H)
    RM, s_d, e_d, z_hd = set_dvsSchedule('RM', subInputs)
    #
    y_uvd = {(u, v, d): RM.addVar(vtype=GRB.BINARY, name='y[%d,%d,%d]' % (u, v, d))
             for u in U for v in V for d in D}
    x_uhvdd = {(u, h, v, d1, d2): RM.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d,%d,%d]' % (u, h, v, d1, d2))
               for u in U for h in H for v in V for d1 in Ds for d2 in Ds}
    a_ud = {(u, d): RM.addVar(vtype=GRB.CONTINUOUS, name='a[%d,%d]' % (u, d))
            for u in U for d in D}
    w_ud = {(u, d): RM.addVar(vtype=GRB.CONTINUOUS, name='w[%d,%d]' % (u, d))
            for u in U for d in D}
    W1 = RM.addVar(vtype=GRB.CONTINUOUS, name='W1')
    RM.update()
    #
    # Define constraints related to time slot scheduling
    #
    subInputs = (H, N, c_i, D, l_d, Di, M1)
    dvsSchedule = (s_d, e_d, z_hd)
    set_ctsScheduleDM(RM, subInputs, dvsSchedule)
    for d in D:
        RM.addConstr(quicksum(z_hd[h, d] for h in H) == p_sd[d], name='processingT[%d]' % d)
        RM.addConstr(s_d[d] + (p_sd[d] - 1) == e_d[d], name='seTS_proT[%d]' % d)
    #
    # Define constraints related to vehicle routing
    #
    for u in U:
        for d in D:
            RM.addConstr(quicksum(y_uvd[u, v, d] for v in V) == 1, name='d2v[%d,%d]' % (u, d))
        for v in V:
            for i in N:
                RM.addConstr(quicksum(y_uvd[u, v, d] for d in Di[i]) <= 1,
                             name='xSameLoc[%d,%d,%d]' % (u, v, i))
        for v in V:
            RM.addConstr(quicksum(x_uhvdd[u, h, v, n0, d] for h in H for d in Ds) == 1,
                         name='DpoFlowO[%d,%d]' % (u, v))
            RM.addConstr(quicksum(x_uhvdd[u, h, v, d, n0] for h in H for d in Ds) == 1,
                         name='DpoFlowI[%d,%d]' % (u, v))
            for d1 in D:
                RM.addConstr(quicksum(x_uhvdd[u, h, v, d1, d2] for h in H for d2 in Ds) == y_uvd[u, v, d1],
                             name='OF_ASG[%d,%d,%d]' % (u, v, d1))
                RM.addConstr(quicksum(x_uhvdd[u, h, v, d2, d1] for h in H for d2 in Ds) == y_uvd[u, v, d1],
                             name='IF_ASG[%d,%d,%d]' % (u, v, d1))
        for d1 in D:
            RM.addConstr(a_ud[u, d1] <= cT * s_d[d1], name='beforeST[%d,%d]' % (u, d1))
            for h in H:
                RM.addConstr(w_ud[u, d1] <= 0 + M2 * (1 - quicksum(x_uhvdd[u, h, v, n0, d1] for v in V)),
                             name='zeroWT[%d,%d,%d]' % (u, h, d1))
                for d2 in D:
                    RM.addConstr(cT * (s_d[d1] + p_sd[d1]) + t_uhij[u][h][l_d[d1]][l_d[d2]] \
                                 <= a_ud[u, d2] + M2 * (1 - quicksum(x_uhvdd[u, h, v, d1, d2] for v in V)),
                                 name='AT_LB2[%d,%d,%d,%d]' % (u, h, d1, d2))
                    RM.addConstr(a_ud[u, d2] \
                                 <= cT * h + t_uhij[u][h][l_d[d1]][l_d[d2]] + M2 * (
                                             1 - quicksum(x_uhvdd[u, h, v, d1, d2] for v in V)),
                                 name='AT_UB2[%d,%d,%d,%d]' % (u, h, d1, d2))
                RM.addConstr(h <= s_d[d1] + p_sd[d1] + M1 * (1 - quicksum(x_uhvdd[u, h, v, d1, d2] for v in V for d2 in D)),
                             name='inTS[%d,%d,%d]' % (u, h, d1))
        for d in D:
            RM.addConstr(cT * s_d[d] - w_ud[u, d] == a_ud[u, d], name='calAT[%d,%d]' % (u, d))
    #
    # Maximum waiting time (epigraph function)
    #
    for u in U:
        for d in D:
            RM.addConstr(w_ud[u, d] <= W1)
    #
    obj = LinExpr()
    obj += W1
    RM.setObjective(obj, GRB.MINIMIZE)
    #
    RM.setParam('Threads', NUM_CORES)
    if etc['logFile']:
        RM.setParam('LogFile', etc['logFile'])
    RM.optimize()
    #
    if RM.status == GRB.Status.INFEASIBLE:
        RM.write('%s.lp' % problemName)
        RM.computeIIS()
        RM.write('%s.ilp' % problemName)

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
            logContents += '\t ObjV: %.3f\n' % RM.objVal
            f.write(logContents)
            f.write('\n')
            f.write('Time slot scheduling\n')
            for d in D:
                f.write('\t D%d: TS [%02d, %02d]\n' % (d, s_d[d].x, e_d[d].x))
            f.write('\n')
            f.write('Vehicle routing for each scenario\n')
            f.write('\n')
            for u in U:
                f.write('Scenario %d \n' % u)
                for v in V:
                    demand = []
                    for d in D:
                        if y_uvd[u, v, d].x > 0.5:
                            demand.append(d)
                    _route = {}
                    for h in H:
                        for d1 in Ds:
                            for d2 in Ds:
                                if x_uhvdd[u, h, v, d1, d2].x > 0.5:
                                    _route[d1] = d2
                    route = [n0, _route[n0]]
                    while route[-1] != n0:
                        route.append(_route[route[-1]])
                    f.write('\t V%d: %s (%s);\n' % (v, str(demand), '->'.join(map(str, route))))
                    f.write('\t\t\t\t\t (%s)\n' % '-'.join(['%.2f' % (cT * s_d[d].x - w_ud[u, d].x) for d in route[1:-1]]))
        #
        # Write a csv file
        #
        with open(etc['solFileCSV'], 'wt') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            header = ['objV', 'eliCpuTime', 'eliWallTime']
            writer.writerow(header)
            writer.writerow([RM.objVal, eliCpuTime, eliWallTime])
        #
        # Write pickle files recording scenario and the optimal solution
        #
        with open(etc['inputFile'], 'wb') as fp:
            pickle.dump(scenario, fp)
        #
        #
        # Write a pickle file recording scenario and the optimal solution
        #
        subInputs = [D, l_d, H]
        dvs = [s_d, e_d, z_hd]
        _s_d, _e_d, _z_hd = get_dvsScheduleVal(subInputs, dvs)
        _y_uvd = {(u, v, d): y_uvd[u, v, d].x for u in U for v in V for d in D}
        _x_uhvdd = {(u, h, v, d1, d2): x_uhvdd[u, h, v, d1, d2].x
                   for u in U for v in V for h in H for d1 in Ds for d2 in Ds}
        _a_ud = {(u, d): a_ud[u, d].x for u in U for d in D}
        _w_ud = {(u, d): w_ud[u, d].x for u in U for d in D}
        _W1 = W1.x
        #
        sol = {'s_d': _s_d, 'e_d': _e_d, 'z_hd': _z_hd,
               #
               'y_uvd': _y_uvd, 'x_uhvdd': _x_uhvdd,
               'a_ud': _a_ud, 'w_ud': _w_ud,
               #
               'W1': _W1}
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
    for fn in os.listdir(target_dpath):
        if not fn.endswith('.pkl'):
            continue
        if 'rb' not in fn: continue
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
        etc = {'inputFile': opath.join(input_dpath, 'input-%s.pkl' % prefix)}
        etc['solFilePKL'] = opath.join(sol_dpath, 'sol-%s.pkl' % prefix)
        etc['solFileCSV'] = opath.join(sol_dpath, 'sol-%s.csv' % prefix)
        etc['solFileTXT'] = opath.join(sol_dpath, 'sol-%s.txt' % prefix)
        etc['logFile'] = opath.join(log_dpath, 'log-%s.txt' % prefix)
        run(scenario_loader(ifpath), etc)



if __name__ == '__main__':
    # from problems import ms_ex0, ms_ex1
    # run(ms_ex0(), writing_files=True)
    # run(ms_ex1(), writing_files=True)

    batch_run()
