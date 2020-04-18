#!/usr/bin/env python
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


from ex09 import node_names, nn2na, C, beq, t, get_selected_arcs, NN

"""
#EX10: Lagrangian Relaxation
1) For #EX11 and T ≤ 8 h𝑠. apply the Lagrangian Relaxation method and find a solution iterating for different values of lagrangian multipliers (λ) between 0 and 1.
2) Plot all the objective function primal solutions for the set of lagrangian multipliers used in 1).
3) What should be the optimum λ related to the shortest path solution?

La idea es agregar un término con langraje
"""

if __name__ == '__main__':
    Aeq, arc_idxs = nn2na(NN)
    bounds = tuple(zip(np.zeros(C.shape[0]), t))
    T = 8

    # generamos un rango entre 0,01 y 1 de 0.01
    λs = np.arange(0.01, 1.0, 0.01, float)

    #bounds = tuple(zip(np.zeros(C.shape[0]), t))
    bounds = tuple(zip(np.zeros(C.shape[0]), np.full(C.shape[0], None)))

    opts = []
    opt_max = (None, -1, None)
    for λ in λs:
        C_monio = C + λ*t
        res = linprog(C_monio,  A_eq=Aeq, b_eq=beq, bounds=bounds, method='simplex')

        opt = res.fun - λ * T
        opts.append(opt)
        if opt_max[1] < opt:
            ruta = get_selected_arcs(arc_idxs, res.x)
            opt_max = (λ, opt, res.x, ruta)


    print(f"Para λ={opt_max[0]:.2f} ")
    print(f"\tLos vértices deben ser {opt_max[2]}")
    print("\tEl camino hallado es: %s" % [(node_names[x[0]], node_names[x[1]]) for x in opt_max[3]])
    distancia_posta = np.dot(opt_max[2],C)   # es el escalar entre los vertices prendidos y los costos de cada vertice
    print(f"\tLa distancia mínima es {distancia_posta:.2f}")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.annotate(f'λ = {opt_max[0]}  Č = {opt_max[1]:.2f}',
                xy=( opt_max[0], opt_max[1]), xytext=(opt_max[0], opt_max[1] + 3 ), horizontalalignment="center",
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    ax.set_ylim(0, 20)
    plt.plot ( λs, opts)
    plt.show()


