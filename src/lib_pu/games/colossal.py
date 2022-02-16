from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import src.lib_pu.games as pu_games


class Colossal(pu_games.Game):
    # For current setting
    n = 25
    cont_shape = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    host_shape = [2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2]
    
    @staticmethod
    def name() -> str:
        return pu_games.COLOSSAL
    
    @staticmethod
    def default_outcome_dist() -> List[float]:
        # uniform distribution
        return [1/Colossal.n for _ in range(Colossal.n)]

    @staticmethod
    def message_structure() -> List[List[int]]:
        return list(Colossal.create_graph().edges)

    @staticmethod
    def cont_default() -> np.ndarray:
        return np.array([[1/shape for _ in range(shape)] for shape in Colossal.cont_shape])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[1/shape for _ in range(shape)] for shape in Colossal.host_shape])
    
    @staticmethod
    def create_graph():
        g = nx.Graph()
        g.add_nodes_from(range(Colossal.n))
        
        # create grid
        for x in range(5):
            for y in range(5):
                curr_node = x + 5*y
                # horizontal edge
                if x != 4 and (y % 2) == 0:
                    g.add_edge(curr_node, curr_node+1)
                # vertical edge
                if y != 4:
                    g.add_edge(curr_node, curr_node+5)
        return g
        
        # draw for debugging
        # nx.draw(g)
        # plt.show()
