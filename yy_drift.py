import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import OptimiserParamSchedule
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential

from ml_genn.compilers.event_prop_compiler import default_params

from argparse import ArgumentParser

import os

from ml_genn.losses import RelativeMeanSquareError


from ml_genn.utils.data import generate_yin_yang_dataset


parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
parser.add_argument("--delays_lr", type=float, default=0.0, help="Learning rate for the delays")
parser.add_argument("--dt", type=float, default=0.005,  help="Timestep")
parser.add_argument("--num_hidden", type=int, default=10, help="Number of hidden neurons")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()

NUM_INPUT = 4
NUM_HIDDEN = args.num_hidden
NUM_OUTPUT = 3
BATCH_SIZE = 150
NUM_EPOCHS = 300
NUM_TRAIN = 5000
NUM_VALID = 1000
early = 1.5
late = 20.0 - early
DT = args.dt


np.random.seed(args.seed)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items())



# Generate Yin-Yang dataset
spikes, labels = generate_yin_yang_dataset(NUM_TRAIN, 
                                           late, early, bias=False)
spikes_val, labels_val = generate_yin_yang_dataset(NUM_VALID, 
                                           late, early, bias=False)
spikes_test, labels_test = generate_yin_yang_dataset(NUM_VALID, 
                                           late, early, bias=False)

network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * NUM_INPUT),
                       NUM_INPUT)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=10000000000),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                        tau_refrac=10000000000,
                                        readout="first_spike_time"), NUM_OUTPUT, record_spikes=True)
    
    in_hid = Connection(input, hidden, Dense(Normal(mean=1.5, sd=0.8), Uniform(-10,-10)),
            Exponential(10.0), max_delay_steps=int(10/DT))

    hid_out = Connection(hidden, output, Dense(Normal(mean=1.5, sd=0.8), Uniform(-10,-10)),
            Exponential(10.0), max_delay_steps=int(10/DT))

    

max_example_timesteps = int(np.ceil((late + 1.5) / DT))
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses=RelativeMeanSquareError(10.0 * 0.2),
                                optimiser=Adam(args.lr), batch_size=BATCH_SIZE,
                                dt=DT, delay_optimiser=Adam(args.delays_lr),
                                delay_learn_conns=[in_hid, hid_out],
                                rng_seed=args.seed)
model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)
results_dic = {}
with compiled_net:
    def alpha_schedule(epoch, alpha):
        return args.lr * (0.9975 ** epoch)
    
    def alpha_schedule_delay(epoch, alpha):
        return args.delays_lr * (0.9975 ** epoch)

    
    

    # Evaluate model on dataset
    callbacks = [OptimiserParamSchedule("alpha", alpha_schedule)]
    if args.delays_lr > 0:
        callbacks.append(OptimiserParamSchedule("alpha", alpha_schedule_delay))
    validation_callbacks = []
    best_acc = 0.0
    for e in range(NUM_EPOCHS):
        _in_hid = compiled_net.connection_populations[in_hid]
        _in_hid.vars["d"].pull_from_device()
        d_view = _in_hid.vars["d"].view.reshape((NUM_INPUT, NUM_HIDDEN))
        print("a", np.min(d_view), np.max(d_view), np.mean(d_view))
        _hid_out = compiled_net.connection_populations[hid_out]
        _hid_out.vars["d"].pull_from_device()
        d_view = _hid_out.vars["d"].view.reshape((NUM_HIDDEN, NUM_OUTPUT))
        print("b", np.min(d_view), np.max(d_view), np.mean(d_view))


        if e % 10 == 0:
            
            print(f"Epoch {e} of {NUM_EPOCHS} best accuracy {best_acc:.4f}")
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input: spikes},
                                            {output: labels},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input: spikes_val}, validation_y={output: labels_val})
        

        if valid_metrics[output].result > best_acc:
            best_acc = valid_metrics[output].result
            results_dic["train_accuracy"] = train_metrics[output].result
            results_dic["valid_accuracy"] = valid_metrics[output].result
            print(f"Epoch {e}: New best accuracy {best_acc:.4f}")
