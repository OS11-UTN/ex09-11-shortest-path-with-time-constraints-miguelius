#!/usr/bin/env python
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


from ex09 import node_names, nn2na, C, beq, t, get_selected_arcs, NN
import matplotlib.colors as mcolors

"""
#EX11: Lagrangian Relaxation with Subgradient Method
1) For #EX11 and T ≤ 8 h𝑠. apply the Lagrangian Relaxation method and find the shortest path iterating between
   several values of lagrangian multipliers using the subgradient method.

USAMOS EL SUBGRADIENTE

El gradiente de L es solo el termino de λ

OBSERVACIONES:

Tardo bastante más. 
"""


def gradiente(i,cuantos):
    c = 100*(i+1)/cuantos
    if c < 4:
        return '#ff0000'
    elif c < 99:
        return '#ffff00'
    return '#00ff00'

if __name__ == '__main__':
    Aeq, arc_idxs = nn2na(NN)
    bounds = tuple(zip(np.zeros(C.shape[0]), t))
    T = 8

    #bounds = tuple(zip(np.zeros(C.shape[0]), t))
    bounds = tuple(zip(np.zeros(C.shape[0]), np.full(C.shape[0], None)))

    λs = []
    opts = []
    opt_max = (None, -1, None)
    epsilon = 0.001
    diff = np.inf
    λiMasUno = 0.001
    i = 1
    while diff > epsilon:
        λ = λiMasUno
        C_monio = C + λ*t
        res = linprog(C_monio,  A_eq=Aeq, b_eq=beq, bounds=bounds, method='revised simplex')
        subgradiente = np.dot(t, res.x) - T
        step = 1/(i**1)
        λiMasUno = λ + step * subgradiente
        diff = abs(λiMasUno - λ)
        λs.append(λ)
        opts.append( np.dot(C_monio, res.x) )
        vertices = res.x
        i+=1
    ruta = get_selected_arcs(arc_idxs, vertices)

    print(f"Para λ={λs[-1]:.2f} ")
    print(f"\tLos vértices deben ser {vertices}")
    print("\tEl camino hallado es: %s" % [(node_names[x[0]], node_names[x[1]]) for x in  ruta])
    distancia_posta = np.dot(vertices,C)   # es el escalar entre los vertices prendidos y los costos de cada vertice
    print(f"\tLa distancia mínima es {distancia_posta:.2f}")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.annotate(f'λ = {λ}  Č = {opts[-1]:.2f}',
                xy=( λ, opts[-1]), xytext=(λ, opts[-1] + 2 ), horizontalalignment="center",
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    ax.set_ylim(0, 12)
    cuantos = len(opts)                      # cuantos λs hallamos
    for i in range(len(opts)):
        dy = 0
        dy += 0.3*(2 - (i % 4))                          # vibore en eje y para no se superpongan los puntos
        plt.scatter(λs[i], opts[i]+dy, c=gradiente(i,cuantos))
    plt.show()

