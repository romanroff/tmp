import time

import networkx as nx
import numpy as np
from tqdm import tqdm, trange

import graph_generator
from common import GraphLayer, CentroidResult, CityResult
from graph_generator import generate_layer, get_node_for_initial_graph_v2
from pfa import find_path


def test_path(
        layer: GraphLayer,
        point_from: int,
        point_to: int
) -> float:
    try:
        my_path = find_path(layer, point_from, point_to)
    except Exception as e:
        print(e)
        return -1
    return my_path[0]


def test_layer(
        points: list[list[int, int]],
        layer: GraphLayer,
) -> tuple[float, list[float]]:
    test_paths: list[float] = []
    start_time = time.time()
    for point_from, point_to in points:
        test_paths.append(test_path(layer, point_from, point_to))
    end_time = time.time()
    test_time = end_time - start_time
    return test_time, test_paths


def get_usual_result(graph: nx.Graph, points: list[tuple[int, int]]) -> tuple[float, list[float]]:
    usual_results: list[float] = []
    start_time = time.time()
    for node_from, node_to in points:
        usual_path = nx.single_source_dijkstra(graph, node_from, node_to, weight='length')
        usual_results.append(usual_path[0])
    end_time = time.time()
    usual_time = end_time - start_time
    return usual_time, usual_results


def get_points(graph: nx.Graph, N: int) -> list[tuple[int, int]]:
    return [get_node_for_initial_graph_v2(graph) for _ in range(N)]


def generate_result(
        usual_results: tuple[float, list[float]],
        test_results: tuple[float, list[float]],
        resolution: float,
        layer: GraphLayer
) -> CentroidResult:
    test_time = test_results[0]
    result = CentroidResult(
        resolution,
        len(layer.centroids_graph.nodes),
        len(layer.centroids_graph.edges),
        len(layer.centroids_graph.nodes) / len(layer.graph.nodes)
    )
    result.speed_up.append(abs(usual_results[0] / test_time))
    result.absolute_time.append(test_time)

    for i, p in enumerate(test_results[1]):
        if p == -1:
            continue
        usual_path_len = usual_results[1][i]
        result.errors.append(abs(usual_path_len - p) / usual_path_len)
        result.absolute_err.append(abs(usual_path_len - p))
    return result


def test_graph(graph: nx.Graph, name: str, city_id: str, points: list[tuple[int, int]] = None,
               resolutions: list[float] = None, pos=2, logs=True) -> CityResult:
    # print(name, nx.is_connected(graph))
    max_alpha = 1 if resolutions is None else max(resolutions)
    delta = max_alpha / 40

    if resolutions is None:
        resolutions = []
        resolutions += [i/10 for i in range(1, 10, 1)]
        resolutions += [i for i in range(1, 10, 1)]
        resolutions += [i for i in range(10, 50, 2)]
        resolutions += [i for i in range(50, 100, 5)]
        resolutions += [i for i in range(100, 500, 10)]
        resolutions += [i for i in range(500, 1000, 50)]
        resolutions += [i for i in range(1000, 2000, 200)]
    if points is None:
        N: int = 1000
        points = [get_node_for_initial_graph_v2(graph) for _ in trange(N, desc='generate points')]
    else:
        N = len(points)

    has_coords = 'x' in [d for u, d in graph.nodes(data=True)][0]

    usual_results = get_usual_result(graph, points)

    result = CityResult(
        name=name,
        name_suffix='',
        city_id=city_id,
        nodes=len(graph.nodes),
        edges=len(graph.edges)
    )

    alphas = set()
    with tqdm(desc=f'resolutions for {name}', position=pos,
              total=max_alpha) as progres:
        prev = 0
        for r in resolutions:
            start = time.time()
            community = graph_generator.resolve_communities(graph, r)
            if len(community) < 10:
                continue
            a = len(community) / len(graph.nodes)
            has = False
            for curr in alphas:
                if abs(curr - a) < delta:
                    has = True
                    break
            if has or a > max_alpha:
                if logs:
                    tqdm.write(f'alpha: {a} -- skip')
                if a == 1 and 1 in alphas or a > max_alpha:
                    break
                else:
                    continue
            alphas.add(a)
            layer, build_communities, build_additional, build_centroid_graph = generate_layer(graph, r,
                                                                                              has_coordinates=has_coords,
                                                                                              communities=community)
            tmp = test_layer(points, layer)
            total = time.time() - start
            text = """
                name:           {}
                alpha:          {:4f}
                total time:     {:.3f}
                prepare time:   {:.3f} 
                    build_communities:      {:.3f}
                    build_additional:       {:.3f}
                    build_centroid_graph:   {:.3f}
                pfa time:       {:.3f}
            """.format(name,a, total, total - tmp[0], build_communities, build_additional, build_centroid_graph, tmp[0])
            if logs:
                tqdm.write(text)
            result.points_results.append(generate_result(usual_results, tmp, r, layer))
            progres.update(a - prev)
            prev = a

        progres.update(max_alpha - prev)

    result.save()
    if logs:
        s = [p.speed_up[0] for p in result.points_results]
        print(np.mean(result.points_results[np.argmax(s)].errors), np.std(result.points_results[0].errors))
        print(np.max(result.points_results[np.argmax(s)].errors))
        print(max(s))
    return result



def get_resolution_for_alpha(graph: nx.Graph, alpha:float ) -> float:
    right_resolution = 5000
    left_resolution = 0.01
    y = len(graph_generator.resolve_communities(graph, (left_resolution+right_resolution)/2)) / len(graph.nodes)
    min_dst = 0.001
    print('start generate resolutions')
    while abs(alpha - y) > min_dst:
        if y > alpha:
            right_resolution = (left_resolution+right_resolution)/2
        else:
            left_resolution = (left_resolution + right_resolution) / 2
        y = len(graph_generator.resolve_communities(graph, (left_resolution + right_resolution) / 2)) / len(graph.nodes)
    return (left_resolution + right_resolution) / 2

