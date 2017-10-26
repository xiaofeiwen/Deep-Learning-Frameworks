import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
import numpy as np
import pandas as pd
tf.set_random_seed(0)
np.random.seed(1)

def load_tf_data():
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    X_train = mnist.train.images
    Y_train = mnist.train.labels
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    return X_train, X_test, Y_train, Y_test

def load_kaggle_data():
    data = pd.read_csv('input/train.csv')
    X = data.iloc[:,1:].values
    X = X.astype(np.float)
    Y = data.iloc[:,0].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2000, random_state=42)
    X_train = X_train.reshape(-1,28,28,1)/255.
    X_test = X_test.reshape(-1,28,28,1)/255.
    Y_train = np.eye(10)[Y_train.reshape(-1)]
    Y_test = np.eye(10)[Y_test.reshape(-1)]
    return X_train, X_test, Y_train, Y_test

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model3(X_train, Y_train, X_test, Y_test,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    t=0
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    m = X_train.shape[0]                          # (n_x: input size, m : number of examples in the train set)
    costs = []                                        # To keep track of the cost
    
    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # correct answers will go here
    Y = tf.placeholder(tf.float32, [None, 10])
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)

    # three convolutional layers with their channel counts, and a
    # fully connected layer (the last layer has 10 softmax neurons)
    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
    
    parameters = {"W1": W1,
                  "B1": B1,
                  "W2": W2,
                  "B2": B2,
                  "W3": W3,
                  "B3": B3,
                  "W4": W4,
                  "B4": B4,
                  "W5": W5,
                  "B5": B5}

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(YY4, W5) + B5

    # Cost function: Add cost function to tensorflow graph
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y))

    # training step, the learning rate is a placeholder
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    # init
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # set learning rate decay
                max_learning_rate = 0.003
                min_learning_rate = 0.0001
                decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-t/decay_speed)
                t += 1
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y,
                                                                            pkeep: 0.75, lr: learning_rate})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, pkeep: 1}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, pkeep: 1}))
        
        
        return parameters

X_train, X_test, Y_train, Y_test = load_kaggle_data()
parameters = model3(X_train, Y_train, X_test, Y_test, num_epochs = 5, minibatch_size = 64)
