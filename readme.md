[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN/ml_genn/master)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/job/ml_genn/job/master/) [![Docs](https://readthedocs.org/projects/ml-genn/badge)](https://ml-genn.readthedocs.io)

# mlGeNN
A library for deep learning with Spiking Neural Networks (SNN)powered by [GeNN](http://genn-team.github.io/genn/), a GPU enhanced Neuronal Network simulation environment.

## Installation
 1. Follow the instructions in https://github.com/genn-team/genn/blob/master/pygenn/README.md to install PyGeNN.
 2. Clone this project
 3. Install mlGeNN with setuptools using ``python ml_genn/setup.py develop`` command
 3. To use mlGeNN to convert ANNs trained with Keras to SNNs, install mlGeNN TF with setuptools using ``python ml_genn_tf/setup.py develop`` command

## Usage
### Convert ANN to SNN
The following example illustrates how to convert an ANN model, defined in Keras, to an SNN using the few-spike ([Stöckl & Maass, 2021](http://dx.doi.org/10.1038/s42256-021-00311-4)) conversion method:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
tf_model = Sequential([
   Conv2D(16, 3, padding='same', activation='relu',
          use_bias=False, input_shape=(32, 32, 3)),
   AveragePooling2D(2),
   Flatten(),
   Dense(10, activation='relu', use_bias=False)])

# compile and train tf_model with TensorFlow
...

from ml_genn_tf.converters import FewSpike

converter = FewSpike(k=8, signed_input=signed_input, norm_data=[x_subset])
net, net_inputs, net_outputs, tf_layer_pops = converter.convert(tf_model)

compiler = converter.create_compiler()
compiled_net = compiler.compile(net, inputs=net_inputs, outputs=net_outputs)

with compiled_net:
    metrics, cb_data = compiled_net.evaluate({net_inputs[0]: validate_x},
                                             {net_outputs[0]: validate_y})
    print(f"Accuracy = {100.0 * metrics[net_outputs[0]].result}%")
```
For further examples, please see the examples/tf folder.

### Training an SNN using eProp
The following example illustrates how to train a simple SNN with the e-prop learning rule ([Bellec, Scherr et al., 2020](http://dx.doi.org/10.1038/s41467-020-17236-y)):
```python
from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import EPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, PoissonInput

# load dataset
...

network = SequentialNetwork()
with network:
    # Populations
    input = InputLayer("poisson_input", 768)
    initial_hidden_weight = 
    hidden = Layer(Dense(Normal(sd=1.0 / np.sqrt(768))),
                   LeakyIntegrateFire(relative_reset=True,
                                      integrate_during_refrac=True),
                   128)
    output = Layer(Dense(Normal(sd=1.0 / np.sqrt(128))),
                   LeakyIntegrate(softmax=True, readout="sum_var"),
                   10)

compiler = EPropCompiler(example_timesteps=200,
                         losses="sparse_categorical_crossentropy",
                         optimiser="adam", batch_size=128)
compiled_net = compiler.compile(network)

with compiled_net:
    metrics, _  = compiled_net.train({input: spikes},
                                     {output: labels},
                                     num_epochs=10, shuffle=True)
    print(f"Accuracy = {100 * metrics[output].result}%")
```