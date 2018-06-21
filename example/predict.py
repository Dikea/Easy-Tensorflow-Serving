# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from easy_tensorflow_serving import ServingClient 


def predict():
    x = np.random.randn(10, 3)
    host_post = "0.0.0.0:9000" 
    model_name = "test_model"
    client = ServingClient(host_post, model_name)
    inputs = {"x": (x[0], tf.float32)}
    outputs = client.predict(inputs)
    print (outputs)
    print (outputs["y"].float_val)


if __name__ == "__main__":
    predict()



