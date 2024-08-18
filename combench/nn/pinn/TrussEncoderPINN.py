import keras
from keras import layers
import tensorflow as tf
import config
import json
import math
import os
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding


# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit


from combench.datasets.TrussPinnDG import TrussPinnDG, dd



save_name = 'EncoderPINN-F4'
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



# --------------------------------------------------------------------------------------------
#       __  __           _      _
#      |  \/  |         | |    | |
#      | \  / | ___   __| | ___| |
#      | |\/| |/ _ \ / _` |/ _ \ |
#      | |  | | (_) | (_| |  __/ |
#      |_|  |_|\___/ \__,_|\___|_|
# --------------------------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="TrussPINN", name="TrussPINN")
class TrussEncoderPINN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dropout = 0.0
        self.design_len = 153 + 1 + 1 + 1 + 1
        self.embed_dim = 64


        # 0: padding
        # 1: 0-bit
        # 2: 1-bit
        # 3: stiffness pool token
        # 4: forces pool token


        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            7,
            self.design_len,
            self.embed_dim,
            mask_zero=True
        )

        self.encoder1 = TransformerEncoder(512, 8)
        self.encoder2 = TransformerEncoder(512, 8)
        self.encoder3 = TransformerEncoder(512, 8)


        # Output Head
        # self.dense_pooling1 = layers.Dense(1, activation='linear')
        self.stiff_pred = layers.Dense(1, activation='linear')

        self.f_pred_h1 = layers.Dense(128, activation='silu')
        self.f_pred = layers.Dense(36, activation='linear')

        self.d_pred_h1 = layers.Dense(128, activation='silu')
        self.d_pred = layers.Dense(36, activation='linear')

        self.k_pred_h1 = layers.Dense(128, activation='silu')
        self.k_pred = layers.Dense(36*36, activation='linear')



    def call(self, inputs, training=None, mask=None):
        x = inputs + 1  # --> shift bits by 1
        # Concat [pool] token which is id 3
        # x has shape (batch_size, seq_len)
        batch_size = tf.shape(x)[0]
        stiff_pool_tokens = tf.ones((batch_size, 1), dtype=tf.int32) + 2
        f_pool_tokens = tf.ones((batch_size, 1), dtype=tf.int32) + 3
        d_pool_tokens = tf.ones((batch_size, 1), dtype=tf.int32) + 4
        k_pool_tokens = tf.ones((batch_size, 1), dtype=tf.int32) + 5

        x = tf.cast(x, tf.int32)
        stiff_pool_tokens = tf.cast(stiff_pool_tokens, tf.int32)
        f_pool_tokens = tf.cast(f_pool_tokens, tf.int32)
        d_pool_tokens = tf.cast(d_pool_tokens, tf.int32)
        k_pool_tokens = tf.cast(k_pool_tokens, tf.int32)

        x = tf.concat([stiff_pool_tokens, f_pool_tokens, d_pool_tokens, x], axis=-1)
        x = self.design_embedding_layer(x, training=training)
        x = self.encoder1(x, training=training)
        x = self.encoder2(x, training=training)

        x = self.encoder3(x, training=training)

        # out1 = self.dense_pooling1(x)
        # out1 = tf.squeeze(out1, axis=-1)
        # get the first sequence element for each batch
        stiff_pool_token_enc = x[:, 0, :]  # (batch_size, embed_dim)
        out1 = self.stiff_pred(stiff_pool_token_enc)

        f_pool_token_enc = x[:, 1, :]  # (batch_size, embed_dim)
        f_pool_token_enc = self.f_pred_h1(f_pool_token_enc)
        out2 = self.f_pred(f_pool_token_enc)

        d_pool_token_enc = x[:, 2, :]  # (batch_size, embed_dim)
        d_pool_token_enc = self.d_pred_h1(d_pool_token_enc)
        out3 = self.d_pred(d_pool_token_enc)

        k_pool_token_enc = x[:, 3, :]  # (batch_size, embed_dim)
        k_pool_token_enc = self.k_pred_h1(k_pool_token_enc)
        out4 = self.k_pred(k_pool_token_enc)

        return out1, out2, out3, out4


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    data_loss_fn = tf.keras.losses.MeanSquaredError()
    data_loss_tracker = tf.keras.metrics.Mean(name="loss")

    force_loss_fn = tf.keras.losses.MeanSquaredError()
    disp_loss_fn = tf.keras.losses.MeanSquaredError()
    k_loss_fn = tf.keras.losses.MeanSquaredError()
    physics_loss_tracker = tf.keras.metrics.Mean(name="physics_loss")



    def train_step(self, data):
        # designs, stiffness, forces = data
        designs, stiffness, forces, displacements, f_full, u_full, k_full = data
        stiffness = stiffness / 1e9

        # Combine
        batch_size = tf.shape(designs)[0]
        k_full = tf.reshape(k_full, (batch_size, 36*36))
        k_full = k_full / 1e10

        with tf.GradientTape() as tape:
            # Predictions
            predictions, f_pred, d_pred, k_pred = self(designs, training=False)
            data_loss = self.data_loss_fn(stiffness, predictions)
            # f_loss = self.force_loss_fn(f_full, f_pred)
            # d_loss = self.disp_loss_fn(u_full, d_pred)
            k_loss = self.k_loss_fn(k_full, k_pred)
            phy_loss = k_loss #+ f_loss + d_loss

            # Loss
            loss = data_loss + phy_loss

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(phy_loss)
        return {"loss": self.data_loss_tracker.result(), 'physics_loss': self.physics_loss_tracker.result()}

    def test_step(self, data):
        # designs, stiffness, forces = data
        designs, stiffness, forces, displacements, f_full, u_full, k_full = data
        stiffness = stiffness / 1e9

        # Combine
        batch_size = tf.shape(designs)[0]
        k_full = tf.reshape(k_full, (batch_size, 36 * 36))
        k_full = k_full / 1e10


        # Predictions
        predictions, f_pred, d_pred, k_pred = self(designs, training=False)
        data_loss = self.data_loss_fn(stiffness, predictions)
        # f_loss = self.force_loss_fn(f_full, f_pred)
        # d_loss = self.disp_loss_fn(u_full, d_pred)
        k_loss = self.k_loss_fn(k_full, k_pred)
        phy_loss = k_loss #+ f_loss + d_loss

        # Loss
        loss = data_loss + phy_loss

        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(phy_loss)
        return {"loss": self.data_loss_tracker.result(), 'physics_loss': self.physics_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.data_loss_tracker, self.physics_loss_tracker]

def get_model(design_len, checkpoint_path=None):
    design_len = design_len

    model = TrussEncoderPINN()
    decisions = tf.zeros((1, design_len))
    model(decisions)

    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()

    return model



if __name__ == '__main__':
    train()










