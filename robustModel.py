import multiprocessing
import time
from gurobipy import *
from basicModel import set_dvsSchedule, set_ctsScheduleDM

NUM_CORES = multiprocessing.cpu_count()


def run(inputs, is_pkl=False):
    startCpuTime, startWallTime = time.clock(), time.time()
    assert len(inputs) == 19
    problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, U, p_ud, t_uhij, M1, M2 = inputs
    #
    RM, g_jd, s_d, e_d, z_hd = set_dvsSchedule('RM', (D, k_i, l_d, H))
    #
    y_uvd = {(u, v, d): RM.addVar(vtype=GRB.BINARY, name='y[%d,%d,%d]' % (u, v, d))
             for u in U for v in V for d in D}
    x_uhvdd = {(u, h, v, d1, d2): RM.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d,%d,%d]' % (u, h, v, d1, d2))
               for u in U for h in H for v in V for d1 in Ds for d2 in Ds}
    a_ud = {(u, d): RM.addVar(vtype=GRB.CONTINUOUS, name='a[%d,%d]' % (u, d))
            for u in U for d in D}
    w_ud = {(u, d): RM.addVar(vtype=GRB.CONTINUOUS, name='w[%d,%d]' % (u, d))
            for u in U for d in D}
    epi_W = RM.addVar(vtype=GRB.CONTINUOUS, name='epi_W')
    RM.update()
    #
    obj = LinExpr()
    obj += epi_W
    RM.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constraints related to time slot scheduling
    #
    subInputs = (D, k_i, l_d, Di, H, N, c_i, T_i, M1)
    dvsSchedule = g_jd, s_d, e_d, z_hd
    set_ctsScheduleDM(RM, subInputs, dvsSchedule)
    for u in U:
        for d in D:
            RM.addConstr(quicksum(z_hd[h, d] for h in H) >= p_ud[u][d], name='processingT[%d,%d]' % (u, d))
            RM.addConstr(s_d[d] + (p_ud[u][d] - 1) <= e_d[d], name='seTS_proT[%d,%d]' % (u, d))
    #
    # Define constraints related to vehicle routing
    #
    for u in U:
        for d in D:
            RM.addConstr(quicksum(y_uvd[u, v, d] for v in V) == 1, name='d2v[%d,%d]' % (u, d))
        for v in V:
            for i in N:
                RM.addConstr(quicksum(y_uvd[u, v, d] for d in Di[i]) <= 1, name='xSameLoc[%d,%d,%d]' % (u, v, i))
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
        for d in D:
            RM.addConstr(0 <= a_ud[u, d], name='initAT1[%d,%d]' % (u, d))
            RM.addConstr(a_ud[u, d] <= M2, name='initAT2[%d,%d]' % (u, d))
            RM.addConstr(a_ud[u, d] <= cT * s_d[d], name='beforeST[%d,%d]' % (u, d))
        for v in V:
            for h in H:
                for d in D:
                    RM.addConstr(cT * s_d[d] - t_uhij[u][h][n0][l_d[d]] <= w_ud[u, d] + M2 * (1 - x_uhvdd[u, h, v, n0, d]),
                                 name='calWT1[%d,%d,%d,%d]' % (u, h, v, d))
                    RM.addConstr(t_uhij[u][h][n0][l_d[d]] + w_ud[u, d] <= a_ud[u, d] + M2 * (1 - x_uhvdd[u, h, v, n0, d]),
                                 name='calAT1[%d,%d,%d,%d]' % (u, h, v, d))
                for d1 in D:
                    for d2 in D:
                        RM.addConstr(cT * (s_d[d2] - (s_d[d1] + p_ud[u][d1])) - t_uhij[u][h][l_d[d1]][l_d[d2]] \
                                     <= w_ud[u, d2] + M2 * (1 - x_uhvdd[u, h, v, d1, d2]),
                                     name='calWT2[%d,%d,%d,%d]' % (h, v, d1, d2))
                        RM.addConstr(cT * (s_d[d1] + p_ud[u][d1]) + t_uhij[u][h][l_d[d1]][l_d[d2]] + w_ud[u, d2] \
                                     <= a_ud[u, d2] + M2 * (1 - x_uhvdd[u, h, v, d1, d2]),
                                     name='calAT2[%d,%d,%d,%d,%d]' % (u, h, v, d1, d2))
    #
    # Maximum waiting time (epigraph function)
    #
    for u in U:
        for d in D:
            RM.addConstr(w_ud[u, d] <= epi_W)

    #
    RM.setParam('Threads', NUM_CORES)
    RM.optimize()
    #
    if RM.status == GRB.Status.INFEASIBLE:
        RM.write('%s.lp' % problemName)
        RM.computeIIS()
        RM.write('%s.ilp' % problemName)
    endCpuTime, endWallTime = time.clock(), time.time()
    eliCpuTime, eliWallTime = endCpuTime - startCpuTime, endWallTime - startWallTime
    #
    # Write a file saving the optimal solution
    #
    with open('%s.txt' % problemName, 'w') as f:
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
    if is_pkl:
        import pickle
        #
        dvs = [g_jd, s_d, e_d, z_hd, y_uvd, x_uhvdd, a_ud, w_ud, epi_W]
        with open('is_%s.pkl' % problemName, 'wb') as fp:
            pickle.dump([inputs, dvs], fp)


if __name__ == '__main__':
    from problems import ms_ex0
    run(ms_ex0(), is_pkl=True)
