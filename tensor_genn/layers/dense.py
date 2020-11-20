from tensor_genn.layers import Layer, DenseConnection
from tensor_genn.layers.neuron_models import if_model

class Dense(Layer):

    def __init__(self, model, params, vars_init, global_params, 
                 name, units, signed_spikes=False):
        super(Dense, self).__init__(model, params, vars_init, 
                                    global_params, name)
        self.units = units
        self.signed_spikes = signed_spikes


    def connect(self, sources):
        connections = [DenseConnection(self.units, self.signed_spikes) 
                       for i in range(len(sources))]
        super(Dense, self).connect(sources, connections)


class IFDense(Dense):

    def __init__(self, name, units, threshold=1.0, signed_spikes=False):
        super(IFDense, self).__init__(
            if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold}, 
            name, units, signed_spikes)


    def set_threshold(self, threshold):
        self.global_params['Vthr'] = threshold

        if self.nrn is not None:
            for batch_i in range(self.tg_model.batch_size):
                nrn = self.nrn[batch_i]
                nrn.extra_global_params['Vthr'].view[:] = threshold
