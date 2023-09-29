import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def minimal_example(observations: int = 1000):
    """

    :return:
    """
    # we generate fake data
    xs = np.random.uniform(-10, 10, size=(observations, 1))
    zs = np.random.uniform(-10, 10, size=(observations, 1))

    print(xs.shape)

    generated_inputs = np.column_stack((xs, zs))
    print(generated_inputs.shape)

    noise = np.random.uniform(-1, 1, (observations, 1))

    generated_targets = 2*xs - 3*zs + 5 + noise

    """
    Tensorflow does not work well with xls or csv - it's tensor based so it wants to get tensors
    we have to transform and store our data in .npz-files
    """
    np.savez('data_npz_files/TF_intro', inputs=generated_inputs, targets=generated_targets)

    # create the model with TF
    training_data = np.load('data_npz_files/TF_intro.npz')
    # print(training_data['inputs'])
    input_size = 2 # xs and zs
    output_size = 1

    # in Tensorflow we must build our model:
    # Sequential()-function, that specifies how the model will be laid down ('stacks layers')
    # takes inputs, applies a simple linear transformation and provides outputs
    # linear combination + output = Layer
    # output = np.dot(inputs, weights) + bias

    # tf.keras.layers.Dense(output size) - takes the inputs provided to the model and calculates the dot product
    # of the inputs and the weights and adds the bias - also applies the activation function (optional)
    model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)])

    # configuring the model for training with the optim. algo. - model.compile(optimizer, loss):
    # SGD - stochastic gradient descent

    model.compile(optimizer='sgd', loss='mean_squared_error')

    # epochs a.k.a. iterations
    model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)

    # extract the weight and biases
    weights = model.layers[0].get_weights()[0]
    print(weights)

    bias = model.layers[0].get_weights()[1]
    print(bias)

    # predictions - batch a.k.a is our data here:
    print(model.predict_on_batch(training_data['inputs']).round(1))

    print(training_data['targets'].round(1))

    # plotting the data:
    plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()


def minimal_example_improved_and_tweaked(observations: int = 1000, learning_rate: float = 0.02, loss: str = 'huber_loss'):
    """

    :return:
    """
    # we generate fake data
    xs = np.random.uniform(-10, 10, size=(observations, 1))
    zs = np.random.uniform(-10, 10, size=(observations, 1))

    print(xs.shape)

    generated_inputs = np.column_stack((xs, zs))
    print(generated_inputs.shape)

    noise = np.random.uniform(-1, 1, (observations, 1))

    generated_targets = 2*xs - 3*zs + 5 + noise

    """
    Tensorflow does not work well with xls or csv - it's tensor based so it wants to get tensors
    we have to transform and store our data in .npz-files
    """
    np.savez('data_npz_files/TF_intro', inputs=generated_inputs, targets=generated_targets)

    # create the model with TF
    training_data = np.load('data_npz_files/TF_intro.npz')
    # print(training_data['inputs'])
    input_size = 2 # xs and zs
    output_size = 1

    # in Tensorflow we must build our model:
    # Sequential()-function, that specifies how the model will be laid down ('stacks layers')
    # takes inputs, applies a simple linear transformation and provides outputs
    # linear combination + output = Layer
    # output = np.dot(inputs, weights) + bias

    # tf.keras.layers.Dense(output size) - takes the inputs provided to the model and calculates the dot product
    # of the inputs and the weights and adds the bias - also applies the activation function (optional)
    """
    That's why we don't need the input_size-var :-)
    """

    # variant 2 - we could set a random kernel_initializer (kernel = weight) and a bias_initializer:
    model = tf.keras.Sequential([tf.keras.layers.Dense(output_size,
                                                       kernel_initializer=tf.random_uniform_initializer(minval=-.1,
                                                                                                        maxval=.1),
                                                       bias_initializer=tf.random_uniform_initializer(minval=-.1,
                                                                                                      maxval=.1))])

    # configuring the model for training with the optim. algo. - model.compile(optimizer, loss):
    # SGD - stochastic gradient descent

    custom_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=custom_optimizer, loss=loss)

    # epochs a.k.a. iterations
    model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)

    # extract the weight and biases
    weights = model.layers[0].get_weights()[0]
    print(weights)

    bias = model.layers[0].get_weights()[1]
    print(bias)

    # predictions - batch a.k.a is our data here:
    print(model.predict_on_batch(training_data['inputs']).round(1))

    print(training_data['targets'].round(1))

    # plotting the data:
    plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()


if __name__ == '__main__':
    minimal_example()
    # losses are e. g.: 'mean_squared_error', 'huber_loss'
    minimal_example_improved_and_tweaked(observations=1000, learning_rate=0.02, loss='huber_loss')
