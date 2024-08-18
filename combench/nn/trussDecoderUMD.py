import keras
from keras import layers
import tensorflow as tf
import os
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from combench.models import truss
from copy import deepcopy
from keras_nlp.src.layers.modeling.position_embedding import PositionEmbedding
from keras_nlp.src.utils.keras_utils import clone_initializer


# ------------------------------------
# Vocabulary
# ------------------------------------

# 0: [pad]
# 1: [start]
# 2: node 1 (node idx 0)
#
# ...
# 19: node 2 (node idx 1)
# 20: node 18 (node idx 17)
# 21: [end]

# Output Mapping: Neurons to Vocabulary
# 0: node 1
# 1: node 2
# ...
# 17: node 18
# 18: [end]
# - Simply shift output idx by +2 to get the corresponding input token

# ------------------------------------
# Config
# ------------------------------------
model_embed_dim = 32

actor_embed_dim = model_embed_dim
actor_heads = 8
actor_dense = 512
actor_dropout = 0.0

critic_embed_dim = model_embed_dim
critic_heads = 8
critic_dense = 512
critic_dropout = 0.0

USE_PROBLEM_ENCODING = True

# ------------------------------------
# Problem
# ------------------------------------

problem = {
    'nodes': [
        [0.0, 0.0], [0.0, 1.5], [0.0, 3.0],
        [1.2, 0.0], [1.2, 1.5], [1.2, 3.0],
        [2.4, 0.0], [2.4, 1.5], [2.4, 3.0],
        # [3.5999999999999996, 0.0], [3.5999999999999996, 1.5], [3.5999999999999996, 3.0],
        # [4.8, 0.0], [4.8, 1.5], [4.8, 3.0],
        # [6.0, 0.0], [6.0, 1.5], [6.0, 3.0]
    ],
    'nodes_dof': [
        [0, 0], [0, 0], [0, 0],
        [1, 1], [1, 1], [1, 1],
        # [1, 1], [1, 1], [1, 1],
        # [1, 1], [1, 1], [1, 1],
        # [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1]
    ],
    'member_radii': 0.2,
    'youngs_modulus': 210000000000.0,
    'load_conds': [
        [
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            # [0, 0], [0, 0], [0, 0],
            # [0, 0], [0, 0], [0, 0],
            # [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, -1]
        ]
    ]
}
max_nodes = len(problem['nodes'])
max_design_len = (max_nodes * (max_nodes - 1)) // 2
truss.set_norms(problem)


# problem2 = deepcopy(problem)
# problem2 = set_load(problem2, 10, [0, -1])
# truss.set_norms(problem2)
#
# problem3 = deepcopy(problem)
# problem3 = set_load(problem3, 9, [0, -1])
# truss.set_norms(problem3)

# from combench.models.truss.problems.cantilever import get_problems
# params = {
#     'x_range': 4,
#     'y_range': 3,
#     'x_res_range': [4, 4],
#     'y_res_range': [3, 3],
#     'radii': 0.2,
#     'y_modulus': 210e9
# }
# train_problems, val_problems, val_problems_out = get_problems(params=params, sample_size=64)
# train_problems = train_problems[:6] + train_problems[7:]
# train_problems = train_problems[1:]


from combench.models.truss.problems.truss_type_1 import TrussType1
N = 3
problem_set = TrussType1.enumerate({
    'x_range': N,
    'y_range': N,
    'x_res': N,
    'y_res': N,
    'radii': 0.2,
    'y_modulus': 210e9
})
split_idx = int(len(problem_set) * 0.8)
train_problems = problem_set[:split_idx]
val_problems = problem_set[split_idx:]
val_problems_out = []
if len(train_problems) > 100:
    train_problems = train_problems[:100]




print('Num Training Problems:', len(train_problems))
print('Num Validation Problems:', len(val_problems))
print('Num Validation Problems Out:', len(val_problems_out))


base_dir = os.path.join(config.plots_dir, 'problems')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
for idx, tp in enumerate(train_problems):
    des = [1 for _ in range(truss.rep.get_num_bits(tp))]
    truss.rep.viz(tp, des, f_name=f"train_{idx}.png", base_dir=base_dir)
    if idx > 5:
        break
for idx, vp in enumerate(val_problems):
    des = [1 for _ in range(truss.rep.get_num_bits(vp))]
    truss.rep.viz(vp, des, f_name=f"val_{idx}.png", base_dir=base_dir)
    if idx > 5:
        break
for idx, vp in enumerate(val_problems_out):
    des = [1 for _ in range(truss.rep.get_num_bits(vp))]
    truss.rep.viz(vp, des, f_name=f"val_out_{idx}.png", base_dir=base_dir)
    if idx > 5:
        break




# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="TrussDecoderUMD", name="TrussDecoderUMD")
class TrussDecoderUMD(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True



        # Variables
        self.vocab_size = max_nodes + 2
        self.vocab_output_size = max_nodes + 1  # [end] token is included
        self.gen_design_seq_length = max_design_len
        self.embed_dim = actor_embed_dim
        self.num_heads = actor_heads
        self.dense_dim = actor_dense

        # Conditioning Vector Positional Encoding
        # self.pos_embedding = SinePositionEncoding(name='positional_encoding')
        self.pos_embedding = PositionEmbedding(
            sequence_length=self.gen_design_seq_length,
            initializer=clone_initializer("uniform"),
            name="position_embedding",
        )

        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=actor_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2', dropout=actor_dropout)
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3', dropout=actor_dropout)
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_4', dropout=actor_dropout)
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_5', dropout=actor_dropout)
        # self.decoder_6 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_6', dropout=actor_dropout)

        # Encoder Stack
        # self.encoder_1 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_1', dropout=actor_dropout)
        # self.encoder_2 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='encoder_2', dropout=actor_dropout)

        # Path Prediction Head
        self.output_modeling_head = layers.Dense(
            self.vocab_output_size,
            # kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights, nodes, enc_attn_msk = inputs

        # batch = tf.shape(nodes)[0]
        # enc_attn_msk_beginning = tf.ones((batch, 1), dtype=tf.float32)
        # enc_attn_msk = tf.cast(enc_attn_msk, tf.float32)
        # enc_attn_msk = tf.concat([enc_attn_msk_beginning, enc_attn_msk], axis=1)
        # enc_attn_msk = tf.cast(enc_attn_msk, tf.bool)

        # 1 Weights: (batch, 1, embed_dim)
        weight_seq = self.add_positional_encoding(weights, nodes, training=training)

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training

    def add_positional_encoding(self, weights, nodes, training=False):
        # nodes = nodes / 10.0

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])
        if USE_PROBLEM_ENCODING is False:
            return weight_seq

        weights_nodes = tf.concat([weight_seq, nodes], axis=1)
        weights_nodes_pos = self.pos_embedding(weights_nodes)
        weights_nodes = weights_nodes + weights_nodes_pos

        return weights_nodes

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





# ------------------------------------
# Critic
# ------------------------------------

@keras.saving.register_keras_serializable(package="TrussDecoderCriticUMD", name="TrussDecoderCriticUMD")
class TrussDecoderCriticUMD(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # Variables
        self.num_objectives = 1
        self.vocab_size = max_nodes + 2
        self.gen_design_seq_length = max_design_len + 1
        self.embed_dim = critic_embed_dim
        self.num_heads = critic_heads
        self.dense_dim = critic_dense

        # Conditioning Vector Positional Encoding
        # self.pos_embedding = SinePositionEncoding(name='sine_pos_embedding')
        self.pos_embedding = PositionEmbedding(
            sequence_length=self.gen_design_seq_length,
            initializer=clone_initializer("uniform"),
            name="position_embedding",
        )


        # Token + Position embedding
        self.design_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=critic_dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2')
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_3')

        # Output Prediction Head
        self.output_modeling_head = layers.Dense(self.num_objectives, name='output_modeling_head')
        self.activation = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        design_sequences, weights, nodes, enc_attn_msk = inputs

        # batch = tf.shape(nodes)[0]
        # enc_attn_msk_beginning = tf.ones((batch, 1), dtype=tf.float32)
        # enc_attn_msk = tf.cast(enc_attn_msk, tf.float32)
        # enc_attn_msk = tf.concat([enc_attn_msk_beginning, enc_attn_msk], axis=1)
        # enc_attn_msk = tf.cast(enc_attn_msk, tf.bool)

        # 1 Weights: (batch, 1, embed_dim)
        weight_seq = self.add_positional_encoding(weights, nodes)

        # 2. Embed design_sequences
        design_sequences_embedded = self.design_embedding_layer(design_sequences, training=training)

        # 3. Decoder Stack
        decoded_design = design_sequences_embedded
        decoded_design = self.decoder_1(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        decoded_design = self.decoder_2(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)
        # decoded_design = self.decoder_3(decoded_design, encoder_sequence=weight_seq, use_causal_mask=True, training=training)

        # 4. Output Prediction Head
        output_prediction_logits = self.output_modeling_head(decoded_design)
        output_prediction = self.activation(output_prediction_logits)

        return output_prediction  # For training

    def add_positional_encoding(self, weights, nodes):
        # nodes = nodes / 10.0

        # Tile conditioning weights across embedding dimension
        weight_seq = tf.expand_dims(weights, axis=-1)
        weight_seq = tf.tile(weight_seq, [1, 1, self.embed_dim])
        if USE_PROBLEM_ENCODING is False:
            return weight_seq

        weights_nodes = tf.concat([weight_seq, nodes], axis=1)
        weights_nodes_pos = self.pos_embedding(weights_nodes)
        weights_nodes = weights_nodes + weights_nodes_pos

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
    design_len = max_design_len
    conditioning_values = 1
    p_nodes = max_nodes
    node_vars = model_embed_dim

    encoder_attn_mask = tf.ones((1, p_nodes))

    actor_model = TrussDecoderUMD()
    decisions = tf.zeros((1, design_len))
    weights = tf.zeros((1, conditioning_values))
    nodes = tf.zeros((1, p_nodes, node_vars))
    actor_model([decisions, weights, nodes, encoder_attn_mask])

    critic_model = TrussDecoderCriticUMD()
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














