import numpy as np
from typing import List

from .segment import Segment, density, maximum_density_subsegment
from .graph import Graph, rand_adjacency_matrix, maximum_density_subgraph

NODES_DEFAULT_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def rand_temporal_graph(
    n_vertices: int = 10,
    upper_bound: int = 5,
    n_timesteps: int = 1,
):
    """_summary_

    Args:
        n_vertices (int, optional): _description_. Defaults to 10.
        upper_bound (int, optional): _description_. Defaults to 5.
        n_timesteps (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    vertices = [*NODES_DEFAULT_NAMES[:n_vertices]]
    return np.array([
        Graph(vertices, rand_adjacency_matrix(n_vertices, upper_bound))
        for _ in range(n_timesteps)
    ])


def make_segment(t_graph: "TemporalGraph", vertices: List[str] = None):
    """_summary_

    Args:
        t_graph (TemporalGraph): _description_
        vertices (List[str], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    vertices = t_graph.vertices if not vertices else vertices

    densities = [
        Graph(vertices, t_graph[i][vertices]).density
        for i in range(t_graph.n_timesteps)
    ]
    weights = np.ones(t_graph.n_timesteps)

    return Segment(densities, weights)


def get_burstiness(
    t_graph: "TemporalGraph",
    t_beg: int,
    t_end: int,
    vertices: List[str] = None,
):
    """_summary_

    Args:
        t_graph (TemporalGraph): _description_
        t_beg (int): _description_
        t_end (int): _description_
        vertices (List[str], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    vertices = t_graph.vertices if not vertices else vertices
    return density(make_segment(t_graph, vertices)[t_beg:t_end + 1])


def find_density_bursting_subgraph(t_graph: "TemporalGraph", theta: int):
    """_summary_

    Args:
        t_graph (TemporalGraph): _description_
        theta (int): _description_

    Returns:
        _type_: _description_
    """
    burstiness = [0]
    n = 1

    graph = np.sum(t_graph[:theta])

    while True:
        # find densest subgraph in graph
        vertices, _ = maximum_density_subgraph(graph)

        # find [t_beg, t_end] such that burstiness(t_graph, t_beg, t_end) is max
        segment = make_segment(t_graph, vertices)
        t_beg, t_end, density = maximum_density_subsegment(segment, theta)
        burstiness.append(density)
        graph = np.sum(t_graph[int(t_beg):int(t_end) + 1])

        # if the burstiness hasn't increased, break
        if burstiness[n - 1] >= burstiness[n]:
            break

        n += 1

    return vertices, int(t_beg), int(t_end)


class TemporalGraph:
    """
    _summary_
    """

    def __init__(self, snapshots: List[Graph], verbose: bool = False):
        if snapshots is None or len(snapshots) == 0:
            raise ValueError
        self._snapshots = snapshots

        if verbose:
            print("# timesteps :", self.n_timesteps)
            print("vertices :", self.vertices)

    @property
    def snapshots(self):
        return self._snapshots

    @property
    def vertices(self):
        return self._snapshots[0].vertices

    @property
    def n_vertices(self):
        return self._snapshots[0].n_vertices

    @property
    def n_timesteps(self):
        return len(self._snapshots)

    def __getitem__(self, keys):
        return self._snapshots[keys]
