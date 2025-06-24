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
    def __init__(self, pop: Population, pre_conns: Sequence[Connection], post_conns: Sequence[Connection]):
        # Get underlying connection
        self._pre_conns = [get_underlying_conn(conn) for conn in pre_conns]
        self._post_conns = [get_underlying_conn(conn) for conn in post_conns]
        self._pop = get_underlying_pop(pop)
        

    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network


    def on_batch_end(self, batch, _):
        if batch > 0:
            pop = self._compiled_network.neuron_populations[self._pop]
            pop.vars["XPosGradient"].pull_from_device()
            xpos_gradients = pop.vars["XPosGradient"].view
            pop.vars["YPosGradient"].pull_from_device()
            ypos_gradients = pop.vars["YPosGradient"].view
            pop.vars["ZPosGradient"].pull_from_device()
            zpos_gradients = pop.vars["ZPosGradient"].view
            pop.vars["XPos"].pull_from_device()
            pop.vars["YPos"].pull_from_device()
            pop.vars["ZPos"].pull_from_device()
            x_pos = pop.vars["XPos"].view
            y_pos = pop.vars["YPos"].view
            z_pos = pop.vars["ZPos"].view
            for _conn in self._pre_conns:
                conn = self._compiled_network.connection_populations[_conn]
                pre_ind = conn.get_sparse_pre_inds()
                conn.vars["d"].pull_from_device()
                conn.vars["DelayGradient"].pull_from_device()
                if conn.matrix_type & SynapseMatrixConnectivity.SPARSE:
                    pre_ind = conn.get_sparse_pre_inds()
                    delays = conn.vars["d"].values + 1e-8
                    delay_gradients = conn.vars["DelayGradient"].values
                    weighted_grad = delay_gradients / delays
                    gradient_pos = np.zeros((xpos_gradients.shape), dtype=delay_gradients.dtype)
                    np.add.at(gradient_pos, (slice(None), pre_ind), weighted_grad)
                else:
                    delays = conn.vars["d"].view.reshape(_conn.source().shape[0], _conn.target().shape[0])  + 1e-8
                    gradients = conn.vars["DelayGradient"].view.reshape(-1, _conn.source().shape[0], _conn.target().shape[0])
                    gradients = gradients / delays
                    gradient_pos = gradients.sum(1)
                xpos_gradients += (x_pos - 1) * gradient_pos
                ypos_gradients += (y_pos - 1) * gradient_pos
                zpos_gradients += (z_pos - 1) * gradient_pos
            for _conn in self._post_conns:
                conn = self._compiled_network.connection_populations[_conn]
                conn.vars["d"].pull_from_device()
                conn.vars["DelayGradient"].pull_from_device()
                if conn.matrix_type & SynapseMatrixConnectivity.SPARSE:
                    post_ind = conn.get_sparse_post_inds()
                    delays = conn.vars["d"].values  + 1e-8
                    delay_gradients = conn.vars["DelayGradient"].values
                    weighted_grad = delay_gradients / delays
                    gradient_pos = np.zeros((xpos_gradients.shape), dtype=delay_gradients.dtype)
                    np.add.at(gradient_pos, (slice(None), post_ind), weighted_grad)
                else:
                    delays = conn.vars["d"].view.reshape(_conn.source().shape[0], _conn.target().shape[0])  + 1e-8
                    gradients = conn.vars["DelayGradient"].view.reshape(-1, _conn.source().shape[0], _conn.target().shape[0])
                    gradients = gradients / delays
                    gradient_pos = gradients.sum(2)
                xpos_gradients += (x_pos - 1) * gradient_pos
                ypos_gradients += (y_pos - 1) * gradient_pos
                zpos_gradients += (z_pos - 1) * gradient_pos
            pop.vars["XPosGradient"].values = xpos_gradients
            pop.vars["XPosGradient"].push_to_device()
            pop.vars["YPosGradient"].values = ypos_gradients
            pop.vars["YPosGradient"].push_to_device()
            pop.vars["ZPosGradient"].values = zpos_gradients
            pop.vars["ZPosGradient"].push_to_device()
        
        
            


