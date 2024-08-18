import keras
from keras import layers
import tensorflow as tf
import config
import math
import json
import os
# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit


from combench.datasets.TrussPinnDG import TrussPinnDG, dd


save_name = 'BaselinePINN5'
data_save_path = os.path.join(config.database_dir, save_name + '.json')
plot_save_path = os.path.join(config.plots_dir, 'pinn', save_name + '.png')
stiff_norm = 1e9
disp_norm = 1e-6
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





@keras.saving.register_keras_serializable(package="BaselinePINN", name="BaselinePINN")
class BaselinePINN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = 0.0


        # MLP
        self.hidden_1 = layers.Dense(512, activation='silu')
        self.hidden_2 = layers.Dense(512, activation='silu')
        self.hidden_3 = layers.Dense(512, activation='silu')

        # Output Layer
        # - Stiffness
        self.output_layer = layers.Dense(1, activation='linear')

        self.disp_layer = layers.Dense(36, activation='linear')
        self.f_layer = layers.Dense(36, activation='linear')
        self.k_pred = layers.Dense(36*36, activation='linear')


    def call(self, inputs, training=False, mask=None):
        # inputs: [design_bits]
        x = inputs
        print(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x_data = self.output_layer(x)
        x_disp = self.disp_layer(x)
        x_force = self.f_layer(x)
        x_k = self.k_pred(x)
        return x_data, x_disp, x_force, x_k

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

    disp_loss_fn = tf.keras.losses.MeanSquaredError()
    force_loss_fn = tf.keras.losses.MeanSquaredError()
    k_loss_fn = tf.keras.losses.MeanSquaredError()

    # physics_loss_fn = tf.keras.losses.MeanSquaredError()
    physics_loss_tracker = tf.keras.metrics.Mean(name="physics_loss")

    def train_step(self, data):
        designs, stiffness, forces, displacements, f_full, u_full, k_full = data
        stiffness = stiffness / stiff_norm
        # u_full = u_full / disp_norm

        f_full = tf.cast(f_full, tf.float32)

        # Combine
        batch_size = tf.shape(designs)[0]
        k_full = tf.reshape(k_full, (batch_size, 36 * 36))
        k_full = k_full / 1e10

        with tf.GradientTape() as tape:
            # Predictions
            preds, preds_disp, preds_forces, preds_k = self(designs, training=True)
            data_loss = self.data_loss_fn(stiffness, preds)
            disp_loss = self.disp_loss_fn(u_full, preds_disp)
            k_loss = self.k_loss_fn(k_full, preds_k)
            f_loss = self.force_loss_fn(f_full, preds_forces)
            physics_loss = disp_loss + f_loss + k_loss
            print('disp loss', disp_loss)

            # Loss
            loss = data_loss + physics_loss
            # loss = data_loss + (physics_loss * 0.1)

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        return {"loss": self.data_loss_tracker.result(), "physics_loss": self.physics_loss_tracker.result()}

    def test_step(self, data):
        designs, stiffness, forces, displacements, f_full, u_full, k_full = data
        stiffness = stiffness / stiff_norm
        # u_full = u_full / disp_norm

        f_full = tf.cast(f_full, tf.float32)

        # Combine
        batch_size = tf.shape(designs)[0]
        k_full = tf.reshape(k_full, (batch_size, 36 * 36))
        k_full = k_full / 1e10

        # Predictions
        preds, preds_disp, preds_forces, preds_k = self(designs, training=False)
        data_loss = self.data_loss_fn(stiffness, preds)
        disp_loss = self.disp_loss_fn(u_full, preds_disp)
        k_loss = self.k_loss_fn(k_full, preds_k)
        f_loss = self.force_loss_fn(f_full, preds_forces)
        physics_loss = disp_loss + f_loss + k_loss

        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        return {"loss": self.data_loss_tracker.result(), "physics_loss": self.physics_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.data_loss_tracker, self.physics_loss_tracker]


def get_model(design_len, checkpoint_path=None):
    design_len = design_len

    model = BaselinePINN()
    decisions = tf.zeros((1, design_len))
    model(decisions)

    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()

    return model


if __name__ == '__main__':
    train()




