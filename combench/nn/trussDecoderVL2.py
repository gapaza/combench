import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
from copy import deepcopy
import math
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding, RotaryEmbedding
from keras.layers import Embedding

from combench.models import truss
from combench.models.truss.TrussModel import TrussModel

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

# self.design_input_layer = layers.Input(shape=(None, None), ragged=True)

@keras.saving.register_keras_serializable(package="TrussDecoder", name="TrussDecoder")
class TrussDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.vocab_size = 4
        self.vocab_output_size = 2  # Include stop token in variable length chromosome
        self.gen_design_seq_length = config.num_vars
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Node embeddings
        self.node_embedding_layer = layers.Dense(self.embed_dim, activation='linear')
        self.sine_pos_embedding = SinePositionEncoding()
        self.node_encoder = TransformerEncoder(self.embed_dim, self.num_heads, name='node_encoder')
        self.node_encoder2 = TransformerEncoder(self.embed_dim, self.num_heads, name='node_encoder2')

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
        design_sequences, weights, nodes, enc_attn_msk = inputs

        batch = tf.shape(nodes)[0]
        enc_attn_msk_beginning = tf.ones((batch, 1), dtype=tf.float32)
        enc_attn_msk = tf.cast(enc_attn_msk, tf.float32)
        enc_attn_msk = tf.concat([enc_attn_msk_beginning, enc_attn_msk], axis=1)
        enc_attn_msk = tf.cast(enc_attn_msk, tf.bool)

        # 1 Weights: (batch, 1, embed_dim)
        weight_seq = self.add_positional_encoding(weights, nodes)

        # 2. Embed design_sequences: (batch, seq_len, embed_dim)
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_5(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)

        # 4. Design Prediction Head
        design_prediction_logits = self.design_prediction_head(decoded_design)
        design_prediction = self.activation(design_prediction_logits)

        return design_prediction

    def add_positional_encoding(self, weights, nodes):
        nodes = nodes / 10.0

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])
        nodes = self.node_embedding_layer(nodes)

        weights_nodes = tf.concat([weight_seq, nodes], axis=1)
        weights_nodes_pos = self.sine_pos_embedding(weights_nodes)
        weights_nodes = weights_nodes + weights_nodes_pos
        weights_nodes = self.node_encoder(weights_nodes)
        weights_nodes = self.node_encoder2(weights_nodes)

        return weights_nodes

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)






    def generate(self, problem, w=[0.5, 0.7]):
        weights = []
        for i in w:
            weights.append([i])
        batch_size = len(weights)
        designs = [[] for x in range(batch_size)]
        start_token_idx = 1
        observation = [[start_token_idx] for x in range(batch_size)]
        num_p_vars = truss.rep.get_num_bits(problem)
        num_nodes = len(problem['nodes'])
        problem_encoding, pad_mask = truss.rep.get_problem_encoding_padded(problem, pad_len=num_nodes)
        problem_encoding = tf.convert_to_tensor(problem_encoding, dtype=tf.float32)
        problem_encoding = tf.expand_dims(problem_encoding, axis=0)
        problem_encoding = tf.tile(problem_encoding, [batch_size, 1, 1])
        pad_mask = tf.convert_to_tensor(pad_mask, dtype=tf.int32)
        pad_mask = tf.expand_dims(pad_mask, axis=0)
        pad_mask = tf.tile(pad_mask, [batch_size, 1])
        for x in range(num_p_vars):
            obs_input = tf.convert_to_tensor(observation, dtype=tf.int32)
            weight_input = tf.convert_to_tensor(weights, dtype=tf.float32)
            pred_probs = self([obs_input, weight_input, problem_encoding, pad_mask], training=False)
            all_token_probs = pred_probs[:, x, :]  # shape (batch, 2)
            all_token_log_probs = tf.math.log(all_token_probs + 1e-10)

            # Soft sampling
            # samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)

            # Top 1 sampling
            samples = tf.argmax(all_token_probs, axis=-1, output_type=tf.int64)
            samples = tf.expand_dims(samples, axis=-1)

            next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
            batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
            next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))
            actions = next_bit_ids  # (batch,)
            for idx, act in enumerate(actions.numpy()):
                m_action = int(deepcopy(act))  # 0, 1, or 2 for end token
                curr_design = designs[idx]
                curr_design.append(m_action)
                observation[idx].append(m_action + 2)

        return designs





# ------------------------------------
# Critic
# ------------------------------------

# Vocabulary
# 0: [pad]
# 1: [start]
# 2: 0-bit
# 3: 1-bit
# 4: [stop]

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

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Node Embedding Layer
        self.node_embedding_layer = layers.Dense(self.embed_dim, activation='linear')
        self.sine_pos_embedding = SinePositionEncoding(name='node_positional_encoding')
        self.node_encoder = TransformerEncoder(self.embed_dim, self.num_heads, name='node_encoder')
        self.node_encoder2 = TransformerEncoder(self.embed_dim, self.num_heads, name='node_encoder2')

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=critic_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights, nodes, enc_attn_msk = inputs

        batch = tf.shape(nodes)[0]
        enc_attn_msk_beginning = tf.ones((batch, 1), dtype=tf.float32)
        enc_attn_msk = tf.cast(enc_attn_msk, tf.float32)
        enc_attn_msk = tf.concat([enc_attn_msk_beginning, enc_attn_msk], axis=1)
        enc_attn_msk = tf.cast(enc_attn_msk, tf.bool)

        # 1.2 Weights: (batch, 1, embed_dim)
        weight_seq = self.add_positional_encoding(weights, nodes)

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_4(decoded_design, encoder_sequence=weight_seq, encoder_padding_mask=enc_attn_msk, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training

    def add_positional_encoding(self, weights, nodes):
        nodes = nodes / 10.0

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])


        nodes = self.node_embedding_layer(nodes)
        weights_nodes = tf.concat([weight_seq, nodes], axis=1)
        weights_nodes_pos = self.sine_pos_embedding(weights_nodes)
        weights_nodes = weights_nodes + weights_nodes_pos
        weights_nodes = self.node_encoder(weights_nodes)
        weights_nodes = self.node_encoder2(weights_nodes)

        return weights_nodes


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
    p_nodes = 25
    node_vars = 6

    encoder_attn_mask = tf.ones((1, p_nodes))

    actor_model = TrussDecoder()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    nodes = tf.zeros((1, p_nodes, node_vars))
    actor_model([decisions, weights, nodes, encoder_attn_mask])

    critic_model = TrussDecoderCritic()
    decisions = tf.zeros((1, design_len + 1))
    weights = tf.zeros((1, conditioning_values))
    nodes = tf.zeros((1, p_nodes, node_vars))
    critic_model([decisions, weights, nodes, encoder_attn_mask])

    # Load Weights
    if checkpoint_path_actor:
        actor_model.load_weights(checkpoint_path_actor).expect_partial()
    if checkpoint_path_critic:
        critic_model.load_weights(checkpoint_path_critic).expect_partial()

    actor_model.summary()

    return actor_model, critic_model












