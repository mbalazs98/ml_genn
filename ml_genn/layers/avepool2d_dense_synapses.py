import numpy as np
from math import ceil
from pygenn.genn_model import create_custom_init_var_snippet_class
from pygenn.genn_model import init_var

from ml_genn.layers import ConnectivityType
from ml_genn.layers.base_synapses import BaseSynapses
from ml_genn.layers.weight_update_models import signed_static_pulse
from ml_genn.layers.helper import _get_param_2d

avepool2d_dense_init = create_custom_init_var_snippet_class(
    'avepool2d_dense',

    param_names=[
        'pool_kh', 'pool_kw',
        'pool_sh', 'pool_sw',
        'pool_ih', 'pool_iw', 'pool_ic',
        'dense_ih', 'dense_iw', 'dense_ic',
        'dense_units',
    ],

    extra_global_params=[
        ('weights', 'scalar*'),
    ],

    var_init_code='''
    const int pool_kh = $(pool_kh), pool_kw = $(pool_kw);
    const int pool_sh = $(pool_sh), pool_sw = $(pool_sw);
    const int pool_ih = $(pool_ih), pool_iw = $(pool_iw), pool_ic = $(pool_ic);

    // Convert presynaptic neuron ID to row, column and channel in pool input
    const int poolInRow = ($(id_pre) / pool_ic) / pool_iw;
    const int poolInCol = ($(id_pre) / pool_ic) % pool_iw;
    const int poolInChan = $(id_pre) % pool_ic;

    // Calculate corresponding pool output
    const int poolOutRow = poolInRow / pool_sh;
    const int poolStrideRow = poolOutRow * pool_sh;
    const int poolOutCol = poolInCol / pool_sw;
    const int poolStrideCol = poolOutCol * pool_sw;

    $(value) = 0.0;
    if ((poolInRow < (poolStrideRow + pool_kh)) && (poolInCol < (poolStrideCol + pool_kw))) {
        const int dense_ih = $(dense_ih), dense_iw = $(dense_iw), dense_ic = $(dense_ic);

        if ((poolOutRow < dense_ih) && (poolOutCol < dense_iw)) {
            const int dense_units = $(dense_units);
            const int dense_in_unit = poolOutRow * (dense_iw * dense_ic) + poolOutCol * (dense_ic) + poolInChan;
            const int dense_out_unit = $(id_post);

            $(value) = $(weights)[
                dense_in_unit * (dense_units) +
                dense_out_unit];
        }
    }
    ''',
)

class AvePool2DDenseSynapses(BaseSynapses):

    def __init__(self, units, pool_size, pool_strides=None, 
                 connectivity_type='procedural'):
        super(AvePool2DDenseSynapses, self).__init__()
        self.units = units
        self.pool_size = _get_param_2d('pool_size', pool_size)
        self.pool_strides = _get_param_2d('pool_strides', pool_strides, default=self.pool_size)
        self.pool_output_shape = None
        self.connectivity_type = ConnectivityType(connectivity_type)
        if self.pool_strides[0] < self.pool_size[0] or self.pool_strides[1] < self.pool_size[1]:
            raise NotImplementedError('pool stride < pool size is not supported')

    def connect(self, source, target):
        super(AvePool2DDenseSynapses, self).connect(source, target)

        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = source.shape
        self.pool_output_shape = (
            ceil(float(pool_ih - pool_kh + 1) / float(pool_sh)),
            ceil(float(pool_iw - pool_kw + 1) / float(pool_sw)),
            pool_ic,
        )
        output_shape = (self.units, )

        if target.shape is None:
            target.shape = output_shape
        elif output_shape != target.shape:
            raise RuntimeError('target layer shape mismatch')

        self.weights = np.empty((np.prod(self.pool_output_shape), self.units), dtype=np.float64)

    def compile(self, mlg_model, name):
        pool_kh, pool_kw = self.pool_size
        pool_sh, pool_sw = self.pool_strides
        pool_ih, pool_iw, pool_ic = self.source().shape
        
        dense_ih, dense_iw, dense_ic = self.pool_output_shape

        wu_var_init = init_var(avepool2d_dense_init, {
            'pool_kh': pool_kh, 'pool_kw': pool_kw,
            'pool_sh': pool_sh, 'pool_sw': pool_sw,
            'pool_ih': pool_ih, 'pool_iw': pool_iw, 'pool_ic': pool_ic,
            'dense_ih': dense_ih, 'dense_iw': dense_iw, 'dense_ic': dense_ic,
            'dense_units': self.units,
        })

        if self.connectivity_type is ConnectivityType.SPARSE:
            conn = 'DENSE_INDIVIDUALG'
        elif self.connectivity_type is ConnectivityType.PROCEDURAL:
            conn = 'DENSE_PROCEDURALG'
        else:
            raise NotImplementedError(
                f'AvePool2DDenseSynapses do not support connectivity type {self.connectivitytype}')

        wu_model = signed_static_pulse if self.source().neurons.signed_spikes else 'StaticPulse'
        wu_var = {'g': wu_var_init}
        wu_var_egp = {'g': {'weights': self.weights.flatten() / (pool_kh * pool_kw)}}

        super(AvePool2DDenseSynapses, self).compile(mlg_model, name, conn, wu_model, {}, wu_var,
                                                    {}, {}, 'DeltaCurr', {}, {}, None, wu_var_egp)
