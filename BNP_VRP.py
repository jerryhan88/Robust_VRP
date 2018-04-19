import multiprocessing
from gurobipy import *

NUM_CORES = multiprocessing.cpu_count()


def run(inputs):
    assert len(inputs) == 11
    n0, H, cT, D, Ds, l_d, Di, p_d, t_hij, s_d, M = inputs
    #
    VRP = Model('VRP')
    x_hdd = {(h, d1, d2): VRP.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d]' % (h, d1, d2))
              for h in H for d1 in Ds for d2 in Ds}
    a_d, w_d = {}, {}
    for d in D:
        a_d[d] = VRP.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % d)
        w_d[d] = VRP.addVar(vtype=GRB.CONTINUOUS, name='w[%d]' % d)
    w_c = VRP.addVar(vtype=GRB.CONTINUOUS, name='w_c')
    VRP.update()
    #
    obj = LinExpr()
    obj += w_c
    VRP.setObjective(obj, GRB.MINIMIZE)
    #
    # Define constraints related to vehicle routing
    #
    VRP.addConstr(quicksum(x_hdd[h, n0, d] for h in H for d in Ds) == 1, name='DpoFlowO')
    VRP.addConstr(quicksum(x_hdd[h, d, n0] for h in H for d in Ds) == 1, name='DpoFlowI')
    for d1 in D:
        VRP.addConstr(quicksum(x_hdd[h, d1, d2] for h in H for d2 in Ds) == 1,
                     name='OF_ASG[%d]' % d1)
        VRP.addConstr(quicksum(x_hdd[h, d2, d1] for h in H for d2 in Ds) == 1,
                     name='IF_ASG[%d]' % d1)
    for d in D:
        VRP.addConstr(a_d[d] <= cT * s_d[d], name='beforeST[%d]' % d)
    for h in H:
        for d in D:
            VRP.addConstr(cT * s_d[d] - t_hij[h][n0][l_d[d]] <= w_d[d] + M * (1 - x_hdd[h, n0, d]),
                         name='calWT1[%d,%d]' % (h, d))
            VRP.addConstr(t_hij[h][n0][l_d[d]] + w_d[d] <= a_d[d] + M * (1 - x_hdd[h, n0, d]),
                         name='calAT1[%d,%d]' % (h, d))
        for d1 in D:
            for d2 in D:
                VRP.addConstr(cT * (s_d[d2] - (s_d[d1] + p_d[d1])) - t_hij[h][l_d[d1]][l_d[d2]] \
                             <= w_d[d2] + M * (1 - x_hdd[h, d1, d2]),
                             name='calWT2[%d,%d,%d]' % (h, d1, d2))
                VRP.addConstr(cT * (s_d[d1] + p_d[d1]) + t_hij[h][l_d[d1]][l_d[d2]] + w_d[d2] \
                             <= a_d[d2] + M * (1 - x_hdd[h, d1, d2]),
                             name='merAT2[%d,%d,%d]' % (h, d1, d2))
    #
    # Maximum waiting time (epigraph function)
    #
    for d in D:
        VRP.addConstr(w_d[d] <= w_c)
    #
    VRP.setParam('OutputFlag', False)
    VRP.setParam('Threads', NUM_CORES)
    VRP.optimize()
    #
    if VRP.status == GRB.Status.INFEASIBLE:
        VRP.write('VRP.lp')
        VRP.computeIIS()
        VRP.write('VRP.ilp')
        assert False

    _route = {}
    for h in H:
        for d1 in Ds:
            for d2 in Ds:
                if x_hdd[h, d1, d2].x > 0.5:
                    _route[d1] = d2
    route = [n0, _route[n0]]
    while route[-1] != n0:
        route.append(_route[route[-1]])
    print('\t %s (%s);\n' % (str(D), '->'.join(map(str, route))))
    print('\t\t\t\t\t (%s)\n' % '-'.join(['%.2f' % (cT * s_d[d] - w_d[d].x) for d in route[1:-1]]))





def test():
    import os.path as opath
    import pickle
    #
    fpath = opath.join('_temp', 'is_s1.pkl')
    with open(fpath, 'rb') as fp:
        inputs, sols = pickle.load(fp)
    problemName, n0, V, H, cT, N, Ns, c_i, k_i, T_i, D, Ds, l_d, Di, p_d, t_hij, M1, M2 = inputs

    target_demands = [2, 6, 9]
    D = [d for d in D if d in target_demands]
    Ds = D + [n0]


    _g_jd, _s_d, _e_d, _z_hd, _y_vd, _x_hvdd, _a_d, _w_d, _epi_W = sols
    #
    inputs = [n0, H, cT, D, Ds, l_d, Di, p_d, t_hij, _s_d, M2]
    run(inputs)

if __name__ == '__main__':
    test()