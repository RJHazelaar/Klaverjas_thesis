import numpy as np
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_small_two_headed_nn(learning_rate):
    # Define the input shape

    # Create the first branch of the network

    input = tf.keras.layers.Input(shape=(299,))

    base_layers = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation="relu")], name="base_layers")(
        input
    )

    value_head = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dense(1, activation="linear")],
        name="value_head",
    )(base_layers)

    policy_head = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation="softmax")], name="policy_head")(
        base_layers
    )

    # Create the two-headed model
    model = tf.keras.models.Model(inputs=input, outputs=[value_head, policy_head])

    # Define how to train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def create_normal_two_headed_nn(learning_rate, l1, l2):
    input = tf.keras.layers.Input(shape=(299,))

    base_layers = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
        ],
        name="base_layers",
    )(input)
    
    value_head = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ],
        name="value_head",
    )(base_layers)

    policy_head = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(32, activation="softmax"),
        ],
        name="policy_head",
    )(base_layers)

    # Create the two-headed model
    model = tf.keras.models.Model(inputs=input, outputs=[value_head, policy_head])

    # Define how to train the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss={'value_head': 'mean_absolute_error', 'policy_head': 'sparse_categorical_crossentropy'}
    )
    return model


def create_simple_nn(learning_rate, l1, l2):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(299, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    # define how to train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.build(input_shape=(1, 299))

    return model


def create_normal_nn(learning_rate, l1, l2):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    # define how to train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.build(input_shape=(1, 299))

    return model

def create_alt_nn(learning_rate, l1, l2):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    # define how to train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.build(input_shape=(1, 331))

    return model

def create_bidding_nn(learning_rate, l1, l2):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )
    # define how to train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="categorical_crossentropy")
    model.build(input_shape=(1, 36))

    return model