import multiprocessing
import time
from gurobipy import *

NUM_CORES = multiprocessing.cpu_count()


def set_dvsSchedule(modelName, subInputs):
    MM = Model(modelName)
    D, k_i, l_d, H = subInputs
    g_jd, s_d, e_d, z_hd = {}, {}, {}, {}
    for d in D:
        for j in k_i[l_d[d]]:
            g_jd[j, d] = MM.addVar(vtype=GRB.BINARY, name='g[%d,%d]' % (j, d))
        s_d[d] = MM.addVar(vtype=GRB.INTEGER, name='s[%d]' % d)
        e_d[d] = MM.addVar(vtype=GRB.INTEGER, name='e[%d]' % d)
    for h in H:
        for d in D:
            z_hd[h, d] = MM.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (h, d))
    #
    return MM, g_jd, s_d, e_d, z_hd


def get_dvsScheduleVal(subInputs, dvs):
    D, k_i, l_d, H = subInputs
    g_jd, s_d, e_d, z_hd = dvs
    #
    _g_jd, _s_d, _e_d, _z_hd = {}, {}, {}, {}
    for d in D:
        for j in k_i[l_d[d]]:
            _g_jd[j, d] = g_jd[j, d].x
        _s_d[d] = s_d[d].x
        _e_d[d] = e_d[d].x
    for h in H:
        for d in D:
            _z_hd[h, d] = z_hd[h, d].x
    #
    return _g_jd, _s_d, _e_d, _z_hd

def set_ctsScheduleDM(MM, subInputs, dvsSchedule):
    #
    # Define deterministic constraints related to time slot scheduling
    #
    D, k_i, l_d, Di, H, N, c_i, T_i, M1 = subInputs
    g_jd, s_d, e_d, z_hd = dvsSchedule
    for d in D:
        MM.addConstr(quicksum(g_jd[j, d] for j in k_i[l_d[d]]) == 1, name='d2tw[%d]' % d)
        for j in k_i[l_d[d]]:
            MM.addConstr(T_i[l_d[d]][j][0] <= s_d[d] + M1 * (1 - g_jd[j, d]), name='tw_alpha[%d,%d]' % (j, d))
            MM.addConstr(e_d[d] <= T_i[l_d[d]][j][1] + M1 * (1 - g_jd[j, d]), name='tw_beta[%d,%d]' % (j, d))
    for h in H:
        for d in D:
            MM.addConstr(s_d[d] <= h + M1 * (1 - z_hd[h, d]), name='startTS[%d,%d]' % (h, d))
            MM.addConstr(h <= e_d[d] + M1 * (1 - z_hd[h, d]), name='endTS[%d,%d]' % (h, d))
    for h in H:
        for i in N:
            MM.addConstr(quicksum(z_hd[h, d] for d in Di[i]) <= c_i[i], name='nodeCap[%d,%d]' % (h, i))


def run(inputs, writing_files=False):
    startCpuTime, startWallTime = time.clock(), time.time()
    assert len(inputs) == 18
    problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2 = inputs
    #
    subInputs = (D, k_i, l_d, H)
    BM, g_jd, s_d, e_d, z_hd = set_dvsSchedule('BM', subInputs)
    #
    y_vd = {(v, d): BM.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (v, d))
            for v in V for d in D}
    x_hvdd = {(h, v, d1, d2): BM.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d,%d]' % (h, v, d1, d2))
              for v in V for h in H for d1 in Ds for d2 in Ds}
    a_d, w_d = {}, {}
    for d in D:
        a_d[d] = BM.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % d)
        w_d[d] = BM.addVar(vtype=GRB.CONTINUOUS, name='w[%d]' % d)
    epi_W = BM.addVar(vtype=GRB.CONTINUOUS, name='epi_W')
    BM.update()
    #
    obj = LinExpr()
    obj += epi_W
    BM.setObjective(obj, GRB.MINIMIZE)
    #
    #
    # Define constraints related to time slot scheduling
    #
    subInputs = (D, k_i, l_d, Di, H, N, c_i, T_i, M1)
    dvsSchedule = g_jd, s_d, e_d, z_hd
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
    for d in D:
        BM.addConstr(a_d[d] <= cT * s_d[d], name='beforeST[%d]' % d)
    for v in V:
        for h in H:
            for d in D:
                BM.addConstr(cT * s_d[d] - t_hij[h][n0][l_d[d]] <= w_d[d] + M2 * (1 - x_hvdd[h, v, n0, d]),
                             name='calWT1[%d,%d,%d]' % (h, v, d))
                BM.addConstr(t_hij[h][n0][l_d[d]] + w_d[d] <= a_d[d] + M2 * (1 - x_hvdd[h, v, n0, d]),
                             name='calAT1[%d,%d,%d]' % (h, v, d))
            for d1 in D:
                for d2 in D:
                    BM.addConstr(cT * (s_d[d2] - (s_d[d1] + p_d[d1])) - t_hij[h][l_d[d1]][l_d[d2]] \
                                 <= w_d[d2] + M2 * (1 - x_hvdd[h, v, d1, d2]),
                                 name='calWT2[%d,%d,%d,%d]' % (h, v, d1, d2))
                    BM.addConstr(cT * (s_d[d1] + p_d[d1]) + t_hij[h][l_d[d1]][l_d[d2]] + w_d[d2] \
                                 <= a_d[d2] + M2 * (1 - x_hvdd[h, v, d1, d2]),
                                 name='merAT2[%d,%d,%d,%d]' % (h, v, d1, d2))
    #
    # Maximum waiting time (epigraph function)
    #
    for d in D:
        BM.addConstr(w_d[d] <= epi_W)
    #
    BM.setParam('Threads', NUM_CORES)
    BM.optimize()
    #
    if BM.status == GRB.Status.INFEASIBLE:
        BM.write('%s.lp' % problemName)
        BM.computeIIS()
        BM.write('%s.ilp' % problemName)
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    if writing_files:
        import os.path as opath
        import os
        import pickle
        temp_dir = '_temp'
        if not opath.exists(temp_dir):
            os.mkdir(temp_dir)
        #
        # Write a text file saving the optimal solution
        #
        with open(opath.join(temp_dir, '%s.txt' % problemName), 'w') as f:
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
                f.write('\t\t\t\t\t (%s)\n' % '-'.join(['%.2f' % (cT * s_d[d].x - w_d[d].x) for d in route[1:-1]]))
        #
        # Write a pickle file recording inputs and the optimal solution
        #
        subInputs = [D, k_i, l_d, H]
        dvs = [g_jd, s_d, e_d, z_hd]
        _g_jd, _s_d, _e_d, _z_hd = get_dvsScheduleVal(subInputs, dvs)
        _y_vd = {(v, d): y_vd[v, d].x for v in V for d in D}
        _x_hvdd = {(h, v, d1, d2): x_hvdd[h, v, d1, d2].x
                  for v in V for h in H for d1 in Ds for d2 in Ds}
        _a_d, _w_d = {}, {}
        for d in D:
            _a_d[d] = a_d[d].x
            _w_d[d] = w_d[d].x
        _epi_W = epi_W.x
        sols = [_g_jd, _s_d, _e_d, _z_hd, _y_vd, _x_hvdd, _a_d, _w_d, _epi_W]
        with open(opath.join(temp_dir, 'is_%s.pkl' % problemName), 'wb') as fp:
            pickle.dump([inputs, sols], fp)


if __name__ == '__main__':
    from problems import s0, s1, s2

    run(s0(), writing_files=True)
    run(s1(), writing_files=True)
    run(s2(), writing_files=True)
