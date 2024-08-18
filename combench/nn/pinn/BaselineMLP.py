import keras
from keras import layers
import tensorflow as tf
import config
import math
import os
import json
# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit


from combench.datasets.TrussPinnDG import TrussPinnDG, dd


save_name = 'BaselineMLP4'
data_save_path = os.path.join(config.database_dir, save_name + '.json')
plot_save_path = os.path.join(config.plots_dir, 'pinn', save_name + '.png')
epochs = 100

def train():


    # 1. Build Model
    model = get_model(153, checkpoint_path=None)
    # model = tf.keras.models.load_model(config.load_path)

    # 2. Get Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    jit_compile = False

    # 3. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)
    model.summary()

    # 4. Get Datasets
    dg = TrussPinnDG(dd)
    train_dataset, val_dataset = dg.load_datasets()

    # 5. Get Checkpoints
    checkpoints = [
        keras.callbacks.ModelCheckpoint(
            dd,
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=False
        )
    ]

    # 6. Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=checkpoints
    )

    # 7. Plot History
    plot_history(history)
    with open(data_save_path, 'w') as f:
        hist_dict = history.history
        hist_dict['Name'] = save_name
        json.dump(hist_dict, f)



def plot_history(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # Set y axis range
    plt.ylim([0, 0.3])
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(plot_save_path)













@keras.saving.register_keras_serializable(package="BaselineMLP", name="BaselineMLP")
class BaselineMLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = 0.0


        # MLP
        self.hidden_1 = layers.Dense(512, activation='silu')
        self.hidden_2 = layers.Dense(512, activation='silu')
        self.hidden_3 = layers.Dense(512, activation='silu')
        # self.hidden_4 = layers.Dense(512, activation='silu')
        # self.hidden_5 = layers.Dense(512, activation='silu')

        # Output Layer
        # - Stiffness
        self.output_layer = layers.Dense(1, activation='linear')


    def call(self, inputs, training=False, mask=None):
        # inputs: [design_bits]
        x = inputs
        print(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        # x = self.hidden_4(x)
        # x = self.hidden_5(x)
        x_data = self.output_layer(x)
        return x_data

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # ---------------------
    # Training Loop
    # ---------------------

    # Token Prediction
    data_loss_fn = tf.keras.losses.MeanSquaredError()
    data_loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        # designs, stiffness, forces = data
        designs, stiffness, forces, displacements, f_full, u_full, k_full = data
        stiffness = stiffness / 1e9

        with tf.GradientTape() as tape:
            # Predictions
            predictions = self(designs, training=True)
            data_loss = self.data_loss_fn(stiffness, predictions)

            # Loss
            loss = data_loss

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.data_loss_tracker.update_state(data_loss)
        return {"loss": self.data_loss_tracker.result()}

    def test_step(self, data):
        # designs, stiffness, forces = data
        designs, stiffness, forces, displacements, f_full, u_full, k_full = data
        stiffness = stiffness / 1e9


        # Predictions
        predictions = self(designs, training=False)
        data_loss = self.data_loss_fn(stiffness, predictions)

        # Loss
        loss = data_loss

        self.data_loss_tracker.update_state(data_loss)
        return {"loss": self.data_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.data_loss_tracker]


def get_model(design_len, checkpoint_path=None):
    design_len = design_len

    model = BaselineMLP()
    decisions = tf.zeros((1, design_len))
    model(decisions)

    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()

    return model








if __name__ == '__main__':
    train()


