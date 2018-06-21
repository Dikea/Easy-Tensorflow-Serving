#-*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from easy_tensorflow_serving import export_model 


def main():

    # Simple model
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    w = tf.get_variable('w', shape = [3,1], initializer = tf.truncated_normal_initializer)
    b = tf.get_variable('b', shape = [1], initializer = tf.zeros_initializer) 
    y = tf.matmul(x, w) + b
    ms_loss = tf.reduce_mean((y - y_)**2)
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(ms_loss)
        
    # Initialize session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Prepare data
    # Let the model learn the equation of y = x1 * 1 + x2 * 2 + x3 * 3
    train_x = np.random.randn(1000, 3)
    train_y = np.sum(train_x * np.array([1,2,3]) + 
        np.random.randn(1000, 3) / 100, axis = 1).reshape(-1, 1)

    # Train model
    train_loss = []
    for _ in range(1000):
        loss, _ = sess.run([ms_loss, train_step], feed_dict={x: train_x, y_: train_y})
        train_loss.append(loss)
    print ("Training done.")

    # Export model.
    export_model(sess, 
                inputs={"x": x}, 
                outputs={"y": y}, 
                export_path="./tmp/model")


if __name__ == "__main__":
    main() 
