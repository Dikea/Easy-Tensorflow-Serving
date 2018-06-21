#-*- coding: utf-8 -*-


import os
import tensorflow as tf


__all__ = ["export_model"]


def export_model(sess, inputs, outputs, export_path, model_version=1):
    """Define serving model.
        
        Args:
            sess: the TensorFlow session from which to 
                save the meta graph and variables.
            inputs: the expected inputs when serving, 
                dictionary format, with {name: tensor}.
            outputs: the expected outputs when serving,
                dictionary format, with {name: tensor}.
            export_path: path to save exported model.

    """
    export_path = os.path.join(tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(model_version)))
    if tf.gfile.Exists(export_path):
        tf.gfile.DeleteRecursively(export_path)
        print ("Remove exists {}.".format(export_path))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    inputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in inputs.items()}
    outputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in outputs.items()}
    input_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs, outputs=outputs, 
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) 
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "signature": input_signature
        },
        legacy_init_op=legacy_init_op) 
    builder.save()
    print ("Exporting model to {} done.".format(export_path))
