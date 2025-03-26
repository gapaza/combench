import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from keras_nlp.layers import RotaryEmbedding

max_cities = 20


# Vocabulary (TSP)
# 0: [pad]
# 1: [start]
# 2: City 1
# 3: City 2
# 4: City 3
# 5: City 4
# 6: City 5

actor_embed_dim = 64
actor_heads = 32
actor_dense = 2048
actor_dropout = 0.0

critic_embed_dim = 64
critic_heads = 32
critic_dense = 2048
critic_dropout = 0.0

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="TspDecoderS", name="TspDecoderS")
class TspDecoderS(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Conditioning Vector Positional Encoding
        # self.coordinate_projection = layers.Dense(self.embed_dim, name='coordinate_projection', activation='linear')
        self.coordinate_projection_1 = layers.Dense(self.embed_dim, name='coordinate_projection_1', activation='relu')
        self.coordinate_projection_2 = layers.Dense(self.embed_dim, name='coordinate_projection_2', activation='relu')
        self.coordinate_projection_3 = layers.Dense(self.embed_dim, name='coordinate_projection_3', activation='linear')
        self.positional_encoding = RotaryEmbedding(name='positional_encoding')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=actor_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2', dropout=actor_dropout)
        self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3', dropout=actor_dropout)

        # Design Prediction Head
        self.design_prediction_head = layers.Dense(
            max_cities,
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        
        
    def call(self, inputs, training=False, mask=None):
        tour_sequences, city_locations, city_padding = inputs

        # 1. Encode both tour sequence and city locations
        tour_sequences, tour_mask = self.encode_city_sequence(tour_sequences)
        city_locations, city_mask = self.encode_city_sequence(city_locations)

        # 3. Decoder Stack
        decoded_tour = tour_sequences
        decoded_tour = self.decoder_1(decoded_tour, encoder_sequence=city_locations, encoder_padding_mask=city_padding, use_causal_mask=True, training=training)
        decoded_tour = self.decoder_2(decoded_tour, encoder_sequence=city_locations, encoder_padding_mask=city_padding, use_causal_mask=True, training=training)
        decoded_tour = self.decoder_3(decoded_tour, encoder_sequence=city_locations, encoder_padding_mask=city_padding, use_causal_mask=True, training=training)

        # 4. Tour Prediction Head
        tour_prediction_logits = self.design_prediction_head(decoded_tour)
        tour_prediction = self.activation(tour_prediction_logits)

        return tour_prediction  # For training

    def encode_city_sequence(self, city_seq):
        # city_seq: (batch_size, num_cities, 2)

        mask = tf.cast(tf.not_equal(city_seq, -1), dtype=tf.bool)
        mask = tf.reduce_all(mask, axis=-1, keepdims=False)

        # city_seq = self.coordinate_projection(city_seq)
        city_seq = self.coordinate_projection_1(city_seq)
        city_seq = self.coordinate_projection_2(city_seq)
        city_seq = self.coordinate_projection_3(city_seq)
        city_seq = self.positional_encoding(city_seq)
        return city_seq, mask

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="TspDecoderSCritic", name="TspDecoderSCritic")
class TspDecoderSCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Conditioning Vector Positional Encoding
        self.positional_encoding = RotaryEmbedding(name='positional_encoding')
        # self.coordinate_projection = layers.Dense(self.embed_dim, name='coordinate_projection', activation='linear')
        self.coordinate_projection_1 = layers.Dense(self.embed_dim, name='coordinate_projection_1', activation='relu')
        self.coordinate_projection_2 = layers.Dense(self.embed_dim, name='coordinate_projection_2', activation='relu')
        self.coordinate_projection_3 = layers.Dense(self.embed_dim, name='coordinate_projection_3', activation='linear')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1')
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        tour_sequences, city_locations, city_padding = inputs

        # 1. Encode both tour sequence and city locations
        tour_sequences, tour_mask = self.encode_city_sequence(tour_sequences)
        city_locations, city_mask = self.encode_city_sequence(city_locations)

        # 3. Decoder Stack
        decoded_tour = tour_sequences
        decoded_tour = self.decoder_1(decoded_tour, encoder_sequence=city_locations, encoder_padding_mask=city_padding, use_causal_mask=True, training=training)
        decoded_tour = self.decoder_2(decoded_tour, encoder_sequence=city_locations, encoder_padding_mask=city_padding, use_causal_mask=True, training=training)
        decoded_tour = self.decoder_3(decoded_tour, encoder_sequence=city_locations, encoder_padding_mask=city_padding, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_tour)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training

    def encode_city_sequence(self, city_seq):
        # city_seq: (batch_size, num_cities, 2)

        mask = tf.cast(tf.not_equal(city_seq, -1), dtype=tf.bool)
        mask = tf.reduce_all(mask, axis=-1, keepdims=False)

        # city_seq = self.coordinate_projection(city_seq)
        city_seq = self.coordinate_projection_1(city_seq)
        city_seq = self.coordinate_projection_2(city_seq)
        city_seq = self.coordinate_projection_3(city_seq)
        city_seq = self.positional_encoding(city_seq)
        return city_seq, mask

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ------------------------------------
# Get
# ------------------------------------

def get_models(checkpoint_path_actor=None, checkpoint_path_critic=None):
    tours = tf.zeros((1, max_cities, 2))
    city_locations = tf.zeros((1, max_cities, 2))
    city_padding = tf.zeros((1, max_cities))

    actor_model = TspDecoderS()
    actor_model([tours, city_locations, city_padding])

    critic_model = TspDecoderSCritic()
    critic_model([tours, city_locations, city_padding])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    return actor_model, critic_model







if __name__ == '__main__':
    actor, critic = get_models()

    num_cities = 9
    tours = tf.zeros((1, num_cities, 2))
    city_locations = tf.zeros((1, num_cities, 2))
    city_padding = tf.ones((1, num_cities))

    tour_prediction = actor([tours, city_locations, city_padding])
    print(tour_prediction)







