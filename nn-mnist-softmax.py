import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data          #importing mnist data into variable mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None,784])                    #variable for each 24x24 digit converted to a 1D array
w = tf.Variable(tf.zeros([784,10]))                                 #weights of neuron
b = tf.Variable(tf.zeros([10]))                                     #biases
init = tf.initialize_all_variables()

y = tf.nn.softmax(tf.matmul(x,w)+b)

y_ = tf.placeholder(tf.float32, shape=[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_x,batch_y = mnist.train.next_batch(100)
    train_data = {x:batch_x, y_:batch_y}

    sess.run(train_step, feed_dict = train_data)

is_correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))                       #finding accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

print ("accuracy : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
prediction=tf.argmax(y,1)                                                   #displaying prediction
print ("predictions : ", is_correct.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}, session=sess))
