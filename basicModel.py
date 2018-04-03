from gurobipy import *


def run(inputs):
    N, c_i, k_i, T_i, H, E, V, t_hij, D, l_d, p_d, n0, Ds = inputs
    M1 = len(H)
    M2 = E * len(H)

    RV = Model('RV')

    o_jd, s_d, e_d, z_hd = {}, {}, {}, {}
    y_vd, x_hvdd, u_d, a_d, w_d = {}, {}, {}, {}, {}
    for d in D:
        for j in k_i[l_d[d]]:
            o_jd[j, d] = RV.addVar(vtype=GRB.BINARY, name='w[%d,%d]' % (j, d))
        s_d[d] = RV.addVar(vtype=GRB.INTEGER, name='s[%d]' % d)
        e_d[d] = RV.addVar(vtype=GRB.INTEGER, name='e[%d]' % d)
        for h in H:
            z_hd[h, d] = RV.addVar(vtype=GRB.BINARY, name='z[%d,%d]' % (h, d))
        for v in V:
            y_vd[v, d] = RV.addVar(vtype=GRB.BINARY, name='y[%d,%d]' % (v, d))
        u_d[d] = RV.addVar(vtype=GRB.INTEGER, name='u[%d]' % d)
        a_d[d] = RV.addVar(vtype=GRB.CONTINUOUS, name='a[%d]' % d)
        w_d[d] = RV.addVar(vtype=GRB.CONTINUOUS, name='w[%d]' % d)
    for h in H:
        for v in V:
            for d1 in Ds:
                for d2 in Ds:
                    x_hvdd[h, v, d1, d2] = RV.addVar(vtype=GRB.BINARY, name='x[%d,%d,%d,%d]' % (h, v, d1, d2))
    RV.update()
    #
    obj = LinExpr()
    for d in D:
        obj += w_d[d]
    RV.setObjective(obj, GRB.MINIMIZE)
    #
    for d in D:
        RV.addConstr(quicksum(o_jd[j, d] for j in k_i[l_d[d]]) == 1, name='d2t[%d]' % d)  # ct2
    for d in D:
        RV.addConstr(s_d[d] + p_d[d] == e_d[d], name='s_e_ts[%d]' % d)  # ct3
        for h in H:
            RV.addConstr(s_d[d] <= h + M1 * (1 - z_hd[h, d]), name='sh_ts[%d,%d]' % (h, d))  # ct4
            RV.addConstr(h <= e_d[d] + M1 * (1 - z_hd[h, d]), name='eh_ts[%d,%d]' % (h, d))  # ct5
    for h in H:
        for i in N:
            RV.addConstr(quicksum(z_hd[h, d] for d in D if l_d[d] == i) <= c_i[i], name='n_c[%d,%d]' % (h, i))  # ct6
    for d in D:
        for j in k_i[l_d[d]]:
            RV.addConstr(T_i[l_d[d]][j][0] <= s_d[d] + M1 * (1 - o_jd[j, d]), name='tw_alpha[%d,%d]' % (j, d))  # ct7
            RV.addConstr(e_d[d] <= T_i[l_d[d]][j][1] + M1 * (1 - o_jd[j, d]), name='tw_beta[%d,%d]' % (j, d))  # ct8
    for v in V:
        for h in H:
            for i in N:
                iLocDemands = [d for d in D if l_d[d] == i]
                for d1 in iLocDemands:
                    for d2 in iLocDemands:
                        RV.addConstr(x_hvdd[h, v, d1, d2] == 0, name='sl_ds[%d,%d,%d,%d]' % (h, v, d1, d2))  # ct9
    for d in D:
        RV.addConstr(quicksum(y_vd[v, d] for v in V) == 1, name='d2v[%d]' % d)  # ct10
    for v in V:
        RV.addConstr(quicksum(x_hvdd[h, v, n0, d] for d in D for h in H) <= 1, name='ofD[%d]' % v)  # ct11_1
        RV.addConstr(quicksum(x_hvdd[h, v, d, n0] for d in D for h in H) <= 1, name='inD[%d]' % v)  # ct11_1
        for d1 in D:
            RV.addConstr(quicksum(x_hvdd[h, v, d1, d2] for d2 in Ds for h in H) == y_vd[v, d1],
                         name='a_of[%d,%d]' % (v, d1))  # ct12_1
            RV.addConstr(quicksum(x_hvdd[h, v, d2, d1] for d2 in Ds for h in H) == y_vd[v, d1],
                         name='a_if[%d,%d]' % (v, d1))  # ct12_2
    for d in D:
        RV.addConstr(1 <= u_d[d], name='se1[%d]' % d)  # ct13_1
        RV.addConstr(u_d[d] <= len(D), name='se2[%d]' % d)  # ct13_2

    for h in H:
        for v in V:
            for d1 in D:
                for d2 in D:
                    RV.addConstr(u_d[d1] - u_d[d2] + 1 <= len(D) * (1 - x_hvdd[h, v, d1, d2]),
                                 name='se3[%d,%d,%d,%d]' % (h, v, d1, d2))  # ct14
    for h in H:
        for v in V:
            for d in D:
                RV.addConstr(t_hij[h][len(N)][l_d[d]] - a_d[d] <= M2 * (1 - x_hvdd[h, v, n0, d]),
                             name='atc1[%d,%d,%d]' % (h, v, d))  # ct15
    for h in H:
        for v in V:
            for d1 in D:
                for d2 in D:
                    RV.addConstr(a_d[d1] + E * p_d[d1] + t_hij[h][l_d[d1]][l_d[d2]] - a_d[d2] <= M2 * (1 - x_hvdd[h, v, d1, d2]),
                                 name='atc2[%d,%d,%d,%d]' % (h, v, d1, d2))  # ct16
    for d in D:
        RV.addConstr(E * s_d[d] - a_d[d] <= w_d[d], name='wt[%d]' % d)  # ct17
    #
    RV.optimize()


if __name__ == '__main__':
    from problems import ex1
    run(ex1())
