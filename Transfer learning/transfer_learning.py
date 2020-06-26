import os

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def transfer_model_summary(model_file_path, input_shape, include_top_true_false,weights_none_other):
    local_weights_file = model_file_path

    pre_trained_model = InceptionV3(input_shape = input_shape, 
                                    include_top = include_top_true_false, 
                                    weights = weights_none_other)

    return pre_trained_model.summary()


def transfer_model_setting(model_file_path, input_shape, include_top_true_false,weights_none_other, last_layer):
    local_weights_file = model_file_path

    pre_trained_model = InceptionV3(input_shape = input_shape, 
                                    include_top = include_top_true_false, 
                                    weights = weights_none_other)

    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False
        
        # pre_trained_model.summary()

    last_layer = pre_trained_model.get_layer(last_layer)
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output
    return last_output



def new_model(last_output, activation, dropout_percentage, activation_output, optimizer, loss_type, metrics, output_unit):

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation= activation)(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(dropout_percentage)(x)                  
    # Add a final sigmoid layer for classification
    x = layers.Dense  (output_unit, activation= activation_output)(x)           

    model = Model( pre_trained_model.input, x) 

    model.compile(optimizer = optimizer, 
                loss = loss_type, 
                metrics = [metrics])

    return model
