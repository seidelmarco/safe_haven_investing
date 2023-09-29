import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


def ml_mnist(buffer_size: int = 10000, batch_size: int = 100, hidden_layer_size: int = 100, num_epochs: int = 5,
             learning_rate: float = 0.001):
    """
    The goal is to write a mnist-algorithm that detects which digit is written. The dataset provides 70,000 images
    of handwritten digits (one digit per image) with the shape 28x28 pixels .
    Since there are only 10 digits (0 - 9), this is a classification problem with 10 classes.
    Our goal would be to build a neural network with 2 hidden layers.
    Todo: homework ... play with the hyperparameters...

    Please pay attention to the time it takes for each epoch to conclude.

    Using the code from the lecture as the basis, fiddle with the hyperparameters of the algorithm.

    1. The *width* (the hidden layer size) of the algorithm. Try a hidden layer size of 200. How does the validation
    accuracy of the model change? What about the time it took the algorithm to train? Can you find a hidden layer size
    that does better?

    2. The *depth* of the algorithm. Add another hidden layer to the algorithm. This is an extremely
    important exercise! How does the validation accuracy change? What about the time it took the algorithm to train?
    Hint: Be careful with the shapes of the weights and the biases.

    3. The *width and depth* of the algorithm. Add as many additional layers as you need to reach 5 hidden layers.
    Moreover, adjust the width of the algorithm as you find suitable. How does the validation accuracy change?
    What about the time it took the algorithm to train?

    4. Fiddle with the activation functions. Try applying sigmoid transformation to both layers. The sigmoid
    activation is given by the string 'sigmoid'.

    5. Fiddle with the activation functions. Try applying a ReLu to the first hidden layer and tanh to the second one.
    The tanh activation is given by the string 'tanh'.

    6. Adjust the batch size. Try a batch size of 10000. How does the required time change? What about the accuracy?

    7. Adjust the batch size. Try a batch size of 1. That's the SGD. How do the time and accuracy change? Is the result
    coherent with the theory?

    8. Adjust the learning rate. Try a value of 0.0001. Does it make a difference?

    9. Adjust the learning rate. Try a value of 0.02. Does it make a difference?

    10. Combine all the methods above and try to reach a validation accuracy of 98.5+ percent.

    Good luck!

    :param learning_rate:
    :param buffer_size:
    :param batch_size:
    :param hidden_layer_size: find the optimum between 200 - 500, 50 units is too simple for the model
    :param num_epochs:
    :return:
    """
    mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    print(mnist_info)

    mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

    # mnist does not provide a validation-set, so we have to split it on our own:
    num_validation_samples = .1 * mnist_info.splits['train'].num_examples
    print('We get this float and need to convert/cast it:', num_validation_samples)

    num_validation_samples = tf.cast(num_validation_samples, tf.int64)

    num_test_samples = tf.cast(mnist_info.splits['test'].num_examples, tf.int64)
    num_train_samples = tf.cast(mnist_info.splits['train'].num_examples, tf.int64)
    print('Train, test, validation:', num_train_samples, num_test_samples, num_validation_samples)

    # We'd like to scale our data in order to make it numerically more stable:
    def scale(image, label):
        """
        our images contain pixels of 0 - 255 integers (total black to clear white)
        for scaling and hence obtaining floats between 0 and 1 we have to divide by 255
        :param image:
        :param label:
        :return:
        """
        image = tf.cast(image, tf.float32)
        # . means we want to get a float
        image /= 255.
        return image, label

    # dataset.map(*function*) applies a custom transformation to a given dataset. It takes as input a
    # function which determines the transformation

    scaled_trained_and_validation_data = mnist_train.map(scale)
    test_data = mnist_test.map(scale)

    # Shuffling the data...

    BUFFER_SIZE = buffer_size

    """
    Note: if buffer_size = 1, no shuffling will actually happen, if buffer_size >= num_samples, shuffling
    will happen at once (uniformly), if 1 < buffer_size < num_samples, we will be optimizing computational power
    """

    shuffled_train_and_validation_data = scaled_trained_and_validation_data.shuffle(BUFFER_SIZE)

    # extract again:
    validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
    train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

    # play with this hyperparameter later...
    BATCH_SIZE = batch_size

    train_data = train_data.batch(BATCH_SIZE)

    # validation is forward propagation - we don't need to batch (it's not very computational expensive), however
    # our moder expects batched inputs thus we just override validation_data with the whole number:
    validation_data = validation_data.batch(num_validation_samples)

    # test_data is also forward propagation...
    test_data = test_data.batch(num_test_samples)

    # our validation data must have the same shape and object properties as the train and test data
    # the mnist-data is iterable and in a 2-tuple format (as_supervised=True)
    # so we have to change validation-data into features-and-targets-style
    # iter() creates an object which can be iterated one at a time - preparing for a loop
    # next() loads the next element of an iterable object - since we have just one batch, we load the inputs and targets

    validation_inputs, validation_targets = next(iter(validation_data))

    """
    THE MODEL

    784 input units (28x28 - flattened), 10 output-nodes, 2 hidden layers with 50 nodes each

    """
    # outline the model

    # input_size is legacy from tensorflow1, where you needed it to populate the tf.placeholder - you
    # won't need it anymore in tf2.0
    input_size = 784
    output_size = 10
    hidden_layer_size = hidden_layer_size  # arbitrary and suboptimal - improve the width in homework

    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='tanh'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='tanh'),
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
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    NUM_EPOCHS = num_epochs

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

    model.fit(train_data,
              callbacks=[early_stopping],
              epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose='auto')

    """
    Test the model:
    We test the final prediction power of our model by running it on the test dataset that the algo has NEVER
    seen before.
    The test is the absolute final instance. Do not test before you are completely done with adjusting the model.
    If you adjust your model after testing, you will start overfitting the test dataset.
    """

    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"""
        Test loss: {test_loss:.2f}. Test accuracy: {test_accuracy*100.:.2f}%.
    """)


if __name__ == "__main__":
    ml_mnist(num_epochs=20, batch_size=100, hidden_layer_size=400)
    print('Test 1: 5 epochs, layer size 100, 2 hidden layers - .9809 accuracy, .9780 val_accuracy')
    print("""
        ****************************
        It is important to realize that fiddling with the hyperparameters overfits the validation data.
        In that cases you should stop early (less epochs).
        ****************************
    """)
    print('Test 2: 5 epochs, layer size 200, 2 hidden layers - .9878 accuracy, .9847 val_accuracy')
    print("""
        Adding another layer to the model: We can see that the accuracy does not necessarily improve. This is
        an important lesson for us. Fiddling with a single hyperparameter may not be enough. Sometimes,
        a deeper net needs to also be wider in order to have higher accuracy. Maybe you need more epochs.
    """)
    print('Test 3: 5 epochs, layer size 200, 3 hidden layers - .9863 accuracy, .9863 val_accuracy')
    print("""
        The deeper the net, the wider you need to build it for obtaining better results.
    """)
    print('Test 4: 5 epochs, layer size 500, 5 hidden layers - .9855 accuracy, .9798 val_accuracy')
    print("""
        In Test 5 and 6 we get an inferior result: The sigmoid does not filter the signals as well as relu,
        but still reaches a respectable result.
        
        ReLu does much better since it "cleans" the noise in the data. If a value is negative, relu filters it out,
        while if it is positive, it takes it into account. For mnist we care only about the intensely black and white
        parts in the images of the digits, so such filtering proves benificial.
    """)
    print('Test 5: 5 epochs, layer size 200, 5 hidden layers, sigmoid - .9651 accuracy, .9682 val_accuracy')
    print('Test 6: 5 epochs, layer size 300, 5 hidden layers, sigmoid - .9665 accuracy, .9662 val_accuracy')
    print('Test 7: 5 epochs, layer size 200, 2 hidden layers, relu and tanh - .9886 accuracy, .9877 val_accuracy')
    print('Test 7: 5 epochs, layer size 300, 2 hidden layers, relu and tanh - .9891 accuracy, .9868 val_accuracy')
    print("""
        Test 8: I changed the activation of the 5th layer to tanh. The result should not be significantly different.
        However, with different width and depth, that may change. I found the best result so far with 5 layers, 
        4 times relu, 1 time tanh and a layer size of 300.
    """)
    print('Test 8: 7 epochs, layer size 300, 5 hidden layers, relu and tanh - .9902 accuracy, .9885 val_accuracy')
    print('Test 9: 100 epochs, layer size 300, 5 hidden layers, batch_size: 50000 - .9959 accuracy, .9903 val_accuracy')
    print("""
            Since the learning rate is lower than normal, we may need to adjust the epochs to give the algo
            enough time to learn.
            Test 9: 30 epochs, layer size 300, 5 hidden layers, batch_size: 100, learning_rate: 0.0001
            - .9998 accuracy, .9993 val_accuracy""")
    print("""Test 10 - solution from the course: 
             all activations are ReLu
             10 epochs, layer size 5000, 10 hidden layers - .9746 accuracy, .9798 val_accuracy""")




