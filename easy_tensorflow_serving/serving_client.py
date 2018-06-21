#-*- coding: utf-8 -*-


import numpy as np
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


__all__ = ["ServingClient"]


class ServingClient(object):
    """Tensorflow serving client."""

    def __init__(self, host_port, model_name):
        "Initialize serving client"
        host, port = host_port.split(":")
        channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = "signature"
    
    def predict(self, inputs, timeout=5.0):
        """Predict
            
            Args:
                inputs: the expectd inputs to get predict value,
                    dictionary format, with {name: (value, dtype)},
                timeout: timeout limitation to client request.

            Returns:
                result: the result of client request.  

        """
        inputs = {k: (np.expand_dims(np.array(v[0]), 0), v[1]) 
                  for k, v in inputs.iteritems()}
        for k, v in inputs.iteritems():
            self.request.inputs[k].CopyFrom(
                tf.make_tensor_proto(v[0], shape=v[0].shape, dtype=v[1]))
        result = self.stub.Predict(self.request, timeout)
        return result.outputs
