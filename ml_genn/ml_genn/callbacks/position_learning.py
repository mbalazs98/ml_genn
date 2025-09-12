import logging
import numpy as np

from itertools import chain

from pygenn import SynapseMatrixConnectivity, VarAccessDim
from typing import Optional, Sequence
from .callback import Callback
from ..utils.network import ConnectionType

from pygenn import get_var_access_dim
from ..utils.network import get_underlying_conn, get_underlying_pop
from ..connection import Connection
from .. population import Population
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class LearnPosition(Callback):
    """
    Args:
        conn:               Synapse population to record from        
    """
    def __init__(self, pop: Population, pre_conns: Sequence[tuple], post_conns: Sequence[tuple], num_dims: int):
        # Get underlying connection
        self._pop = get_underlying_pop(pop)
        self._pre_conns = [get_underlying_conn(conn[0]) for conn in pre_conns]
        self._pre_pops = [get_underlying_pop(pop[1]) for pop in pre_conns]
        self._post_conns = [get_underlying_conn(conn[0]) for conn in post_conns]
        self._post_pops = [get_underlying_pop(pop[1]) for pop in post_conns]
        self._num_dims = num_dims


    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network


    def on_batch_end(self, batch, _):
        if batch > 0:
            pop = self._compiled_network.neuron_populations[self._pop]
            pos_gradients, positions = [], []
            for i in range(self._num_dims):
                pop.vars["PosGradient"+str(i)].pull_from_device()
                pos_gradients.append(pop.vars["PosGradient"+str(i)].view)
                pop.vars["Pos"+str(i)].pull_from_device()
                positions.append(pop.vars["Pos"+str(i)].view)
            for _conn, _pop in zip(self._pre_conns, self._pre_pops):
                conn = self._compiled_network.connection_populations[_conn]
                pre_pop = self._compiled_network.neuron_populations[_pop]
                pre_pos = []
                for i in range(self._num_dims):
                    pre_pop.vars["Pos"+str(i)].pull_from_device()
                    pre_pos.append(pre_pop.vars["Pos"+str(i)].view)
                conn.vars["d"].pull_from_device()
                conn.vars["DelayGradient"].pull_from_device()
                if conn.matrix_type & SynapseMatrixConnectivity.SPARSE:
                    pre_ind = conn.get_sparse_pre_inds()
                    post_ind = conn.get_sparse_post_inds()
                    delays = conn.vars["d"].values + 1e-8
                    gradients = conn.vars["DelayGradient"].values
                    for i in range(self._num_dims):
                        gradient_pos = np.zeros_like(pos_gradients[i])
                        np.add.at(
                            gradient_pos,
                            (np.arange(gradients.shape[0])[:, None], post_ind[None, :]),
                            gradients / delays * (positions[i][post_ind] - pre_pos[i][pre_ind])
                        )
                        pos_gradients[i] += gradient_pos
                else:
                    delays = conn.vars["d"].view.reshape(_conn.source().shape[0], _conn.target().shape[0])  + 1e-8
                    gradients = conn.vars["DelayGradient"].view.reshape(-1, _conn.source().shape[0], _conn.target().shape[0])
                    for i in range(self._num_dims):
                        pos_gradients[i] += (gradients / delays * (positions[i][:, None] - pre_pos[i][None, :])).sum(1)
            for _conn, _pop in zip(self._post_conns, self._post_pops):
                conn = self._compiled_network.connection_populations[_conn]
                post_pop = self._compiled_network.neuron_populations[_pop]
                post_pos = []
                for i in range(self._num_dims):
                    post_pop.vars["Pos"+str(i)].pull_from_device()
                    post_pos.append(post_pop.vars["Pos"+str(i)].view)
                conn.vars["d"].pull_from_device()
                conn.vars["DelayGradient"].pull_from_device()
                if conn.matrix_type & SynapseMatrixConnectivity.SPARSE:
                    pre_ind = conn.get_sparse_pre_inds()
                    post_ind = conn.get_sparse_post_inds()
                    delays = conn.vars["d"].values  + 1e-8
                    gradients = conn.vars["DelayGradient"].values
                    for i in range(self._num_dims):
                        gradient_pos = np.zeros_like(pos_gradients[i])
                        np.add.at(
                            gradient_pos,
                            (np.arange(gradients.shape[0])[:, None], pre_ind[None, :]),
                            gradients / delays * (positions[i][pre_ind] - post_pos[i][post_ind])
                        )
                        pos_gradients[i] += gradient_pos
                else:
                    delays = conn.vars["d"].view.reshape(_conn.source().shape[0], _conn.target().shape[0])  + 1e-8
                    gradients = conn.vars["DelayGradient"].view.reshape(-1, _conn.source().shape[0], _conn.target().shape[0])
                    pos_gradients[i] += (gradients / delays * (positions[i][:, None] - post_pos[i][None, :])).sum(2)
            for i in range(self._num_dims):
                pop.vars["PosGradient"+str(i)].values = pos_gradients[i]
                pop.vars["PosGradient"+str(i)].push_to_device()
