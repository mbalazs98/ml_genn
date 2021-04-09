import tensorflow as tf
from tensorflow.keras import models, layers, datasets
from ml_genn import Model
from ml_genn.converters import RateBased, FewSpike
from ml_genn.norm import DataNorm, SpikeNorm
from ml_genn.utils import parse_arguments, raster_plot
import numpy as np
from six import iteritems
from time import perf_counter


if __name__ == '__main__':
    args = parse_arguments('Simple CNN classifier model')
    print('arguments: ' + str(vars(args)))

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Retrieve and normalise MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train[:args.n_train_samples].reshape((-1, 28, 28, 1)) / 255.0
    y_train = y_train[:args.n_train_samples]
    x_test = x_test[:args.n_test_samples].reshape((-1, 28, 28, 1)) / 255.0
    y_test = y_test[:args.n_test_samples]
    x_norm = x_train[np.random.choice(x_train.shape[0], args.n_norm_samples, replace=False)]

    # Create, train and evaluate TensorFlow model
    tf_model = models.Sequential([
        layers.Conv2D(16, 5, padding='valid', activation='relu', use_bias=False, input_shape=x_train.shape[1:]),
        layers.AveragePooling2D(2),
        layers.Conv2D(8, 5, padding='valid', activation='relu', use_bias=False),
        layers.AveragePooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', use_bias=False),
        layers.Dense(64, activation='relu', use_bias=False),
        layers.Dense(y_train.max() + 1, activation='softmax', use_bias=False),
    ], name='simple_cnn')

    if args.reuse_tf_model:
        tf_model = models.load_model('simple_cnn_tf_model')
    else:
        tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tf_model.fit(x_train, y_train, epochs=10)
        models.save_model(tf_model, 'simple_cnn_tf_model', save_format='h5')

    tf_eval_start_time = perf_counter()
    tf_model.evaluate(x_test, y_test)
    print("TF evaluation:%f" % (perf_counter() - tf_eval_start_time))

    # Create, normalise and evaluate ML GeNN model
    converter = FewSpike() if args.few_spike else RateBased(args.input_type)
    mlg_model = Model.convert_tf_model(tf_model, converter=converter, connectivity_type=args.connectivity_type)
    mlg_model.compile(dt=args.dt, batch_size=args.batch_size,
                      rng_seed=args.rng_seed, kernel_profiling=args.kernel_profiling)

    if not args.few_spike:
        if args.norm_method == 'data-norm':
            norm = DataNorm([x_norm], tf_model)
            norm.normalize(mlg_model)
        elif args.norm_method == 'spike-norm':
            norm = SpikeNorm([x_norm])
            norm.normalize(mlg_model, 500)

    time = 10 if args.few_spike else 500
    mlg_eval_start_time = perf_counter()
    acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], time, save_samples=args.save_samples)
    print("MLG evaluation:%f" % (perf_counter() - mlg_eval_start_time))

    if args.kernel_profiling:
        print("Kernel profiling:")
        for n, t in iteritems(mlg_model.get_kernel_times()):
            print("\t%s: %fs" % (n, t))

    # Report ML GeNN model results
    print('Accuracy of SimpleCNN GeNN model: {}%'.format(acc[0]))
    if args.plot:
        neurons = [l.neurons.nrn for l in mlg_model.layers]
        raster_plot(spk_i, spk_t, neurons, time=time)
