import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

import seaborn as sns
sns.set()


def minimal_example(learning_rate: float = 0.02, observations: int = 1000, iterations: int = 100):
    """
    Train the model part - game plan for each iteration:
    - calculate outputs
    - compare outputs to targets through the l2-norm loss
    - print the loss
    - adjust weights and biases
    - repeat
    :param learning_rate:
    :param observations:
    :param iterations: aka epochs
    :return:
    """


    # generate random input data to train on:
    observations = observations

    # input-variables x and z to feed the algorithm
    # uniform draws a random value from the interval, where each number has an equal chance to be selected
    # size = n * k (variables)
    xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
    zs = np.random.uniform(low=-10, high=10, size=(observations, 1))
    # print(xs)

    # let's combine it in one matrix with the shape of n * k ( 1000 * 2)
    inputs = np.column_stack((xs, zs))
    print(inputs.shape)

    # our targets are defined by a function f(x,z) = 2*x - 3*z +5 + some noise
    # 2 and -3 are the weights, the bias is 5 (that's the correct result)

    noise = np.random.uniform(-1, 1, (observations, 1))

    # targets are a linear combination of two vectors 1000 by 1 , a scalar and a noise-vector 1000 by 1
    targets = 2*xs - 3*zs + 5 + noise
    print(targets.shape)

    # plot the training data
    # the point is to see that there is a strong trend that our model should learn to reproduce

    # In order to use 3D-plot, the objects should have a certain shape, so we reshape the targets.
    # The proper method to use is reshape and takes as arguments the dimensions in which we want to fit the object.
    # Edit: it doesn't work this way - so I commented the next line:
    # targets = targets.reshape(observations, )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zs, targets)
    ax.set_xlabel('xs')
    ax.set_ylabel('zs')
    ax.set_zlabel('Targets')

    # you can fiddle with the azim-parameter to plot the data from different angles. Just play around...
    ax.view_init(azim=50)
    plt.show()
    # targets = targets.reshape(observations, 1)

    # initializing the variables:
    # we don't want to start from any arbitrary number like in theory, rather, we randomly select
    # some small initial weigths

    init_range = .1
    # size of the weights-vector is 2x1
    weights = np.random.uniform(-init_range, init_range, size=(2, 1))

    # bias is a scalar
    biases = np.random.uniform(-init_range, init_range, size=1)
    print(weights)
    print(biases)

    # set a learning rate
    # 0.02 seems to be useful - play around with the learning rate...
    eta = learning_rate

    # Train the model:
    for i in range(iterations):
        outputs = np.dot(inputs, weights) + biases
        deltas = outputs - targets

        # L2-norm loss formula: sum of (y - t)**2
        # / observations - we want to obtain the average/mean loss
        loss = np.sum(deltas ** 2) / 2 / observations
        # print(loss)

        deltas_scaled = deltas / observations

        # update rule - following the gradient descent methodology
        weights = weights - eta * np.dot(inputs.T, deltas_scaled)
        biases = biases - eta * np.sum(deltas_scaled)

    # print weights and biases of the last iteration:
    print(f'Last loss: {loss}')
    print(f'The sought weights were 2 and -3, the learned weights are: {weights}')
    print(f'The sought bias is 5 + some noise, the learned bias is: {biases}')

    # plot last outputs vs targets:
    plt.plot(outputs, targets)
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()


if __name__ == '__main__':
    minimal_example(learning_rate=0.05, observations=1000, iterations=250)
    print(tf.__version__)
