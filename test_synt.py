import argparse
import math
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.spatial import KDTree as kd
from tqdm import trange, tqdm

import city_tests
import graph_generator


def gen(N):
    W, H = 1000, 1000
    Q = nx.Graph()
    for i in range(N):
        x = random.random() * W
        y = random.random() * H
        Q.add_node(i, x=x, y=y)

    m = 100000
    for u, d in Q.nodes(data=True):
        for j, t in Q.nodes(data=True):
            if u == j:
                continue
            dd = (d['x'] - t['x']) ** 2 + (d['y'] - t['y']) ** 2
            m = min(m, dd)
    # print('min distance', m)
    return Q


def add_density(H: nx.Graph, r) -> nx.Graph:
    _G = H.copy()
    ids = [node for node in H.nodes()]
    points = [[d['x'], d['y']] for u, d in H.nodes(data=True)]

    tree = kd(points)
    random.seed(123)
    prob = r - int(r)
    for u, du in H.nodes(data=True):

        dists, n_ids = tree.query([du['x'], du['y']], math.ceil(r))
        if type(n_ids) is np.int64:
            n_ids = [n_ids]
            dists = [dists]
        if math.ceil(r) == 1:
            total = len(n_ids)
        else:
            total = len(n_ids) - 1
            if random.random() < prob:
                total += 1
        for i in range(total):
            _id = n_ids[i]
            d = dists[i]
            if ids[_id] == u:
                continue
            _G.add_edge(u, ids[_id], length=d)
    if not nx.is_connected(_G):
        # print('fix connected')
        tmp = []
        for n in nx.connected_components(_G):
            for q in n:
                tmp.append(q)
                break
        for i in range(len(tmp) - 1):
            d1 = _G.nodes[tmp[i]]
            d2 = _G.nodes[tmp[i + 1]]
            _G.add_edge(tmp[i], tmp[i + 1], length=((d1['x'] - d2['x']) ** 2 + (d1['y'] - d2['y']) ** 2) ** 0.5)
    return _G


def calculate(data):
    G = data[0]
    dens = data[1]
    points = data[2]
    NUMBER = data[3]
    THREADS = data[4]
    N = len(G.nodes)
    for d in dens:
        k = d * (N - 1)
        Q = add_density(G, k)
        for u in Q.nodes:
            if u in Q[u]:
                Q.remove_edge(u, u)
        city_tests.test_graph(Q,
                              f'PlanePoints_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                              '0',
                              points=points, pos=NUMBER)
        NUMBER += THREADS


if __name__ == '__main__':
    total = 1
    points_number = 500
    if len(sys.argv) == 2:
        total = int(sys.argv[1])
    else:
        total = int(sys.argv[1])
        points_number = int(sys.argv[2])

    print('THREADS:', total)
    print('POINTS:', points_number)

    dens = [0.0022981490745372685]
    while dens[-1] * 1.6 < 1:
        dens.append(dens[-1] * 1.3)
    dens.append(1)
    dens = np.array(dens)
    dens = dens[dens < 0.05]

    NODES = [2000, 5000, 10000, 15000, 20000, 30000, 50000]

    with Pool(total) as p:
        for j, N in enumerate(NODES):
            G = gen(N)

            points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
                      range(points_number)]

            total_len = len(dens)
            data = [[G, dens[i: total_len: total], points, j * total_len + (i + 1), total] for i in range(total)]
            p.map(calculate, data)
