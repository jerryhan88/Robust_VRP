

def ex1():
    N = list(range(5))
    c_i = [1, 1, 1, 2, 2]
    k_i = map(list, map(range, [1, 1, 2, 2, 1]))
    T_i = [
            [(0, 6)],
            [(6, 12)],
            [(0, 3), (6, 10)],
            [(3, 6), (9, 12)],
            [(3, 11)],
          ]

    H = list(range(12))
    E = 1
    V = list(range(3))
    D = list(range(10))
    l_d = [0, 1, 2, 3, 4,
           0, 3, 3, 4, 4]
    p_d = [1, 1, 2, 2, 1,
           2, 2, 3, 3, 4]

    n0 = len(D)
    Ds = D + [n0]
    t_hij = []
    for _ in H:
        t_ij = []
        for i in N + [n0]:
            t_ij.append([1 if i != j else 0 for j in N + [n0]])
        t_hij.append(t_ij)

    return N, c_i, k_i, T_i, H, E, V, t_hij, D, l_d, p_d, n0, Ds


if __name__ == '__main__':
    ex1()