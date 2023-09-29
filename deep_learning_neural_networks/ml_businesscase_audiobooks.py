import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import tensorflow as tf


def ml_audiobooks(buffer_size: int = 10000, batch_size: int = 100, hidden_layer_size: int = 100, num_epochs: int = 5,
                  learning_rate: float = 0.001):
    """
    Each customer has made a purchase at least once, the algo should predict, if the customer will buy again.

    Wie kamen die wenigen 1er targets (erneuter Kauf) bei über 14.000 gesammelten Erstkäufen zu Stande?
    Die gesammelten Daten (Features) stammen aus 2 Jahren - danach hat man weitere 6 Monate gewartet, ob
    die Erstkunden erneut kaufen. Die Einsen in der Target-Spalte sind die Wiederkäufer.

    Aufgrund der features: wie hoch ist die Wahrscheinlichkeit, dass ein Audiobookkäufer wieder kauft?
    Die Targets sind 1 und 0.
    :return:
    """

    # preprocess the data, balance the dataset
    raw_csv_data = np.loadtxt('data_csv_excel/Audiobooks_data.csv', delimiter=',', dtype=float)
    print(raw_csv_data)

    # nur die features/inputs rausslicen:
    unscaled_inputs_all = raw_csv_data[:, 1:-1]
    #print(unscaled_inputs_all)
    targets_all = raw_csv_data[:, -1]
    count = 0
    for i in targets_all:
        count += 1

    print(count)
    print(targets_all)

    """
    Shuffle the data:
    shape[0] sind die Indices - wir shuffeln nur die Zähler; also immer Partionen des Datasets :-)
    
    We shuffle the indices before balancing, however, we still have to shuffle them AFTER we balance the dataset
    as otherwise, all targets that are 1s will be contained in the train_targets.
    
    We record the variables in themselves, so we don't amend the code that follows.
    """
    shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
    np.random.shuffle(shuffled_indices)
    print('Shuffled inputs:', shuffled_indices)

    # Use the shuffled indices to shuffle the inputs and targets:
    unscaled_inputs_all = unscaled_inputs_all[shuffled_indices] # das ist ein slice - wir legen einfach nur eine
    # Schablone auf alle inputs... :-)
    targets_all = targets_all[shuffled_indices]

    """
    Balancing the dataset: wir müssen unserem Algo die Chance geben, richtig zu lernen, deshalb müssen die Targets
    des gesamten Datasets ausbalanciert sein - also sogenannte "priors" (Vorabwahrscheinlichkeiten) von 0.5 und 0.5
    für ein Zwei-Klassen-Klassifizierungs-Datenset. Sollten wir ein Datenset mit 80% einer Klasse haben, dann lernt
    auch ein "dummer" Algo, dass er immer diese Klasse voraussagen muss und bringt eine sehr hohe accuracy. Ein sehr
    guter Algo könnte mit einer sehr schlechten accuracy abschneiden, obwohl er bei einem ausbalancierten Datenset
    hervorragend voraussagen würde.
    PS: Daran könnte mein Coterra-Algo gescheitert sein...
    """
    num_one_targets = int(np.sum(targets_all))
    print(num_one_targets)
    # I'll just keep as many 0s as there are 1s
    zero_targets_counter = 0
    indices_to_remove = []

    print('Shape all targets - so zählt man in numpy:', targets_all.shape)

    # the shape of targets_all on axis=0 is basically the length of the vector:
    for i in range(targets_all.shape[0]):
        if targets_all[i] == 0:
            zero_targets_counter += 1
            if zero_targets_counter > num_one_targets:
                # hier sammeln wir all die Nullen, die 2237 überschreiten
                indices_to_remove.append(i)

    unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
    targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
    print(targets_equal_priors)
    print(np.count_nonzero(targets_equal_priors))

    """
    Standardize the inputs
    Scaled inputs improve the accuracy by around 10 percent:
    """
    scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

    # shuffle the data (since we are batching, because of it could be that original dataset was ordered by date etc.
    # inside batches the data could be very homogeneous (day of the week effect, promotions, black swan incidents)
    # between batches the data is very heterogeneous
    # that would confuse the stochastic gradient descent
    """
    Examples np.arange()
    
    >>> np.arange(10)  # Stop is 10, start is 0, and step is 1!
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    
    >>> y = np.arange(5, dtype=float)
    >>> y
    array([0., 1., 2., 3., 4.])
    """
    # soooo ... scaled_inputs.shape[0] ist die Anzahl und dient als Stoppwert, Inkrement ist 1
    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to shuffle the inputs and targets:
    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = targets_equal_priors[shuffled_indices]

    """
    Split the data into train, validation and test
    """
    samples_count = shuffled_inputs.shape[0]

    train_samples_count = int(0.8 * samples_count)
    validation_samples_count = int(0.1 * samples_count)
    test_samples_count = samples_count - train_samples_count - validation_samples_count

    # let's extract them from the big dataset:
    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]

    validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
    validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

    test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
    test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

    print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
    print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
    print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

    """
    We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation and test were taken
    from a shuffled dataset. Check if they are balanced too. Note, that each time you rerun the code, you will
    get different values, as each time they are shuffled randomly. If you rerun this whole script, the npzs will
    be overwritten with your newly preprocessed data.
    
    NOTE!!! It's not required to call them inputs= and targets= - these are kwargs, so we can call them mickey mouse :-)
    """

    np.savez('data_npz_files/Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
    np.savez('data_npz_files/Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
    np.savez('data_npz_files/Audiobooks_data_test', inputs=test_inputs, targets=test_targets)

    """
    Crucial to realize!
    Each time we run the code in this sheet, we will preprocess the data once again -
    forgetting the previous preprocessing.
    """

    """
    Create the algorithm:
    """
    # load data
    npz = np.load('data_npz_files/Audiobooks_data_train.npz')
    # standardized data must be floats:
    # Todo: NOTE!!! It's not required to call them inputs= and targets= -
    #  these are kwargs, so we can call them mickey mouse :-)
    train_inputs = npz['inputs'].astype(float)
    # targets must be int since of sparse_categorical_crossentropy for one-hot encoding
    train_targets = npz['targets'].astype(int)

    npz = np.load('data_npz_files/Audiobooks_data_validation.npz')
    validation_inputs, validation_targets = npz['inputs'].astype(float), npz['targets'].astype(int)

    npz = np.load('data_npz_files/Audiobooks_data_test.npz')
    test_inputs, test_targets = npz['inputs'].astype(float), npz['targets'].astype(int)
    print(test_inputs)

    """ Model: outline, optimizers, loss, early stopping and training """

    # just reuse the code from ml_mnist.py and tweak it:

    # input_size is legacy from tensorflow1, where you needed it to populate the tf.placeholder - you
    # won't need it anymore in tf2.0
    input_size = 10
    output_size = 2
    hidden_layer_size = hidden_layer_size  # arbitrary and suboptimal - improve the width in homework

    """
    tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    It takes several arguments, but the most important for us are the hidden_layer_size and the activation function.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    """
    Choose the optimizer and the loss function:
    'categorical_crossentropy' - expects that you have already one-hot encoded the targets
    'sparse_categorical_crossentropy' - applies one-hot encoding in this step (very convenient)
    """

    # Build a custom optimizer for future fiddling:
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # what can you pass to metrics? Look in documentation of the compile()-method:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    """
    Training:
    What will happen inside an epoch:
    1. At the beginning of each epoch, the training loss will set to 0
    2. The algorithm will iterate over a preset number of batches, all from train_data
    3. The weights and biases will be updated as many times as there are batches
    4. We will get a value for the loss-function, indicating how the training is going
    5. We wil see a training accuracy
    6. At the end of the epoch, the algorithm will forward propagate the whole validation set
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

    model.fit(train_inputs, train_targets, batch_size=batch_size,
              epochs=num_epochs,
              callbacks=[early_stopping],
              validation_data=(validation_inputs, validation_targets), verbose='auto')

    """ ACHTUNG BABY! After running the code we see that we start to overfit very early.
        We need to implement an early stopping mechanism - in tf it's called "Callbacks"
        Module: tf.keras.callbacks
        Callbacks are utilities called at certain points during model training.
        e. g. class BaseLogger (accumulates epoch averages), CSVLogger (streams epoch results to a csv,
        class EarlyStopping: stop training when a monitored quantity has stopped improving.
        class LambdaCallback: creating simple, custom callbacks on-the-fly
        class TensorBoard: Tensorboard basic visualizations
    """

    """
    Testing:
    """
    test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
    print(f'\nTest loss: {test_loss:.2f}, test accuracy: {test_accuracy*100.:.2f}%')

    predictions = model.predict(test_inputs, verbose='auto')


if __name__ == '__main__':
    ml_audiobooks(batch_size=10, num_epochs=100, hidden_layer_size=100, learning_rate=0.0001)

