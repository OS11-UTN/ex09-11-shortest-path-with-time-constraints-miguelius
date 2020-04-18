#!/usr/bin/env python
import numpy as np
from pprint import pprint
from scipy.optimize import linprog

"""
#EX09: direct LP approach
The shortest path between node s and t has to be found. Each arc has a distance 
but also the time that it takes to travel between its corresponding vertices.

1) If a person had to travel between s and t in less than 9 hours (T). 
   What’s the shortest path? Try to solve the problem with a simple LP model.
2) What if the maximum available time that this person has drops to 8 hours? 
   What’s the new shortest path? Understand the LP model outputs.
3) What’s the first solution that comes to your mind in order to solve point 2 issues? 
   Is it feasible in reality?
"""

def nn2na(NN):
  idx = np.argwhere(NN)
  NA = np.zeros([NN.shape[0], idx.shape[0]]).astype(int)
  for i, arc in enumerate(idx):
    NA[arc[0], i] = 1
    NA[arc[1], i] = -1

  arc_idx = [ (arc[0], arc[1]) for arc in idx]
  return NA, arc_idx

def get_usage(arc_idxs, use, max_flow):
  return [f"{x} -> {np.round(use[i])} / {max_flow[i]}" for i, x in enumerate(arc_idxs)]

def min_cut(arc_idxs, use, max_flow):
  return list(filter(lambda x: x is not None,
                     [x if max_flow[i] != None and np.isclose(use[i], max_flow[i]) == [True] else None for i, x in
                      enumerate(arc_idxs)]))

def get_selected_arcs(arc_idxs, selected_arcs):
  arc = []
  for idx, i in enumerate(selected_arcs):
      if round(i) == 1:
          arc.append(arc_idxs[idx])
  return arc

def get_arcs_as_tuple_list(NN):
    return [ tuple(x) for x in (np.transpose(np.nonzero(NN)).tolist())]

NN = np.array([
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
])

t=np.array([3,1,3,1,3,3,5])
C = np.array([2, 1, 2, 5, 2, 1, 2])
beq = np.array([1, 0, 0, 0, 0, -1])
node_names = ['s', '2', '3', '4', '5', 't']

if __name__ == '__main__':
    Aeq, arc_idxs = nn2na(NN)
    bounds = tuple(zip(np.zeros(C.shape[0]), t))
    Ts = [9, 8]
    for T in Ts:
        res = linprog(C, A_eq=Aeq, b_eq=beq, bounds=bounds, b_ub=[T],A_ub=[t],  method='revised simplex')

        ruta = get_selected_arcs(arc_idxs, res.x)

        print(f"Para T={T}")
        print(f"\tLos vértices deben ser {res.x}")
        print("\tEl camino hallado es: %s" % [(node_names[x[0]], node_names[x[1]]) for x in ruta])
        for x in res.x:
            if x!= 0 and x!=1:
                print("\t*** ERROR: El camino no parte de s y no llega a t, luego no es válido. :( ***")
                print("\tEx10 plantea relajar este caso, por lo cual la respuesta del punto 3 sería relajandolo")
                break
        print(f"\tLa distancia mínima es {res.fun}")

