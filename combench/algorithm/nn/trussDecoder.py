import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit

actor_embed_dim = 32
actor_heads = 16
actor_dense = 512
actor_dropout = 0.0

critic_embed_dim = 32
critic_heads = 16
critic_dense = 512
critic_dropout = 0.0

# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="TrussDecoder", name="TrussDecoder")
class TrussDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.vocab_size = 4
        self.vocab_output_size = 2
        self.gen_design_seq_length = config.num_vars
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Conditioning Vector Positional Encoding
        self.positional_encoding = SinePositionEncoding(name='positional_encoding')

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )
        
        # Node Embedding Layer
        # self.node_hidden_1 = layers.Dense(self.embed_dim, activation='relu')
        self.node_embedding_layer = layers.Dense(self.embed_dim, activation='linear')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=actor_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2', dropout=actor_dropout)
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3', dropout=actor_dropout)
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4', dropout=actor_dropout)
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5', dropout=actor_dropout)

        # Design Prediction Head
        self.design_prediction_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        design_sequences, weights, nodes = inputs

        # print('Design Sequences:', design_sequences.shape)
        # print('Weights:', weights.shape)
        # print('Nodes:', nodes.shape)

        # 1.1 Embed nodes: (batch, 9, embed_dim)
        # nodes = self.node_hidden_1(nodes)
        nodes = self.node_embedding_layer(nodes)

        # 1.2 Weights: (batch, 1, embed_dim)
        weight_seq = self.add_positional_encoding(weights, nodes)



        # 2. Embed design_sequences: (batch, seq_len, embed_dim)
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True)

        # 4. Design Prediction Head
        design_prediction_logits = self.design_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction  # For training

    def add_positional_encoding(self, weights, nodes):

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])
        weight_seq = tf.concat([weight_seq, nodes], axis=1)

        # For sine positional encoding
        pos_enc = self.positional_encoding(weight_seq)
        weight_seq = weight_seq + pos_enc

        return weight_seq

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="TrussDecoderCritic", name="TrussDecoderCritic")
class TrussDecoderCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = 4
        self.gen_design_seq_length = config.num_vars + 1
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Conditioning Vector Positional Encoding
        self.positional_encoding = SinePositionEncoding(name='positional_encoding')

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Node Embedding Layer
        # self.node_hidden_1 = layers.Dense(self.embed_dim, activation='relu')
        self.node_embedding_layer = layers.Dense(self.embed_dim, activation='linear')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=critic_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights, nodes = inputs

        # 1.1 Embed nodes: (batch, 9, embed_dim)
        # nodes = self.node_hidden_1(nodes)
        nodes = self.node_embedding_layer(nodes)

        # 1.2 Weights: (batch, 1, embed_dim)
        weight_seq = self.add_positional_encoding(weights, nodes)

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training

    def add_positional_encoding(self, weights, nodes):
        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])
        weight_seq = tf.concat([weight_seq, nodes], axis=1)

        # For sine positional encoding
        pos_enc = self.positional_encoding(weight_seq)
        weight_seq = weight_seq + pos_enc

        return weight_seq

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
    design_len = config.num_vars
    conditioning_values = 1
    p_nodes = 9

    actor_model = TrussDecoder()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    nodes = tf.zeros((1, p_nodes, 2))
    actor_model([decisions, weights, nodes])

    critic_model = TrussDecoderCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    nodes = tf.zeros((1, p_nodes, 2))
    critic_model([decisions, weights, nodes])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    return actor_model, critic_model












