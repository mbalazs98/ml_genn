import logging
import numpy as np

from itertools import chain

from pygenn import SynapseMatrixConnectivity
from .callback import Callback
from ..utils.network import ConnectionType

from ..utils.network import get_underlying_conn, get_underlying_pop
from ..connection import Connection
from .. population import Population
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class DeriveDelay(Callback):
    """
    Args:
        conn:               Synapse population to record from        
    """
    def __init__(self, conn: Connection, pop1: Population, pop2: Population, num_dims: int, max_delay: int):
        # Get underlying connection
        self._conn = get_underlying_conn(conn)
        self._pop1 = get_underlying_pop(pop1)
        self._pop2 = get_underlying_pop(pop2)
        self._num_dims = num_dims
        self._max_delay = max_delay


    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network
        

    
    #NOTE that only works for fully connected recurrent populations!
    def enforce_max_distance(self, points):
        center = points.mean(axis=0)
        r = self._max_delay / (2 * np.sqrt(2.0))

        vecs = points - center  # (N, D)

        # ℓ∞ projection: clip coordinates
        vecs = np.clip(vecs, -r, r)

        return center + vecs
    
    def on_batch_end(self, batch, _):
        if batch > 0:
            pop = self._compiled_network.neuron_populations[self._pop1]
            positions = []
            for dim in range(self._num_dims):
                pop.vars["Pos"+str(dim)].pull_from_device()
                positions.append(pop.vars["Pos"+str(dim)].view)
            points1 = np.column_stack(positions)
            if self._max_delay is not None:
                points1 = self.enforce_max_distance(points1)
            pop = self._compiled_network.neuron_populations[self._pop2]
            positions = []
            for dim in range(self._num_dims):
                pop.vars["Pos"+str(dim)].pull_from_device()
                positions.append(pop.vars["Pos"+str(dim)].view)
            points2 = np.column_stack(positions)
            dist = cdist(points1, points2)
            conn = self._compiled_network.connection_populations[self._conn]
            if conn.matrix_type & SynapseMatrixConnectivity.SPARSE:
                conn.vars["d"].pull_from_device()
                conn.get_sparse_pre_inds()
                conn.get_sparse_post_inds()
                dist[conn.get_sparse_pre_inds(), conn.get_sparse_post_inds()].flatten()
                conn.vars["d"].values = dist[conn.get_sparse_pre_inds(), conn.get_sparse_post_inds()].flatten()
            else:
                conn.vars["d"].pull_from_device()
                conn.vars["d"].values = dist.flatten()
            conn.vars["d"].push_to_device()
            if self._max_delay is not None:
                for dim in range(self._num_dims):
                    pop.vars[f"Pos{dim}"].values = points1[:, dim]
                    pop.vars[f"Pos{dim}"].push_to_device()
