import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_eager_execution()


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        # print("I'm here!")
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys
        # print("I'm here2!")
        # print("input shape:", np.shape(K), np.shape(Q))
        # print("query size n window size", window_size_queries, window_size_keys)

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        # print("I'm here3!")
        # print(np.shape(mask_vals))
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        # print("I'm here4!")
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])
        # print("I'm here5!")

        atten_score = tf.matmul(Q, K, transpose_b=True) 
        # print("atten score, atten mask")
        # print(np.shape(atten_score), np.shape(atten_mask))
        if self.use_mask == True:
            atten_score += atten_mask
        attention_weights = tf.nn.softmax(atten_score / np.sqrt(K.get_shape()[2]), axis=-1) #window_size_keys
        # print("I'm here6!")
        return attention_weights


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention
        self.K = self.add_weight(name = "k", shape=[input_size, output_size])
        self.Q = self.add_weight(name = "q", shape=[input_size, output_size])
        self.V = self.add_weight(name = "v", shape=[input_size, output_size])
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        # print("I'm here!")
        # print("Shapes:", np.shape(inputs_for_keys), np.shape(inputs_for_values), np.shape(inputs_for_queries))
        K = tf.tensordot(inputs_for_keys, self.K, axes = 1)
        V = tf.tensordot(inputs_for_values, self.V, axes = 1)
        Q = tf.tensordot(inputs_for_queries, self.Q, axes = 1)
        # print("I'm here 2!")

        attn_matrix = self.attn_mtx([K, Q])
        return tf.matmul(attn_matrix, V)


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.use_mask = use_mask
        self.attention_head1 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.attention_head2 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.attention_head3 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.dense_res = tf.keras.layers.Dense(self.emb_sz)
        

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        k1, v1, q1 = inputs_for_keys, inputs_for_values, inputs_for_queries 
        k2, v2, q2 = inputs_for_keys, inputs_for_values, inputs_for_queries  
        k3, v3, q3 = inputs_for_keys, inputs_for_values, inputs_for_queries  
        res1 = self.attention_head1(k1, v1, q1)
        res2 = self.attention_head2(k2, v2, q2)
        res3 = self.attention_head3(k3, v3, q3)
        combined = tf.concat([res1, res2, res3], axis=-1)
        return self.dense_res(combined)
        


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.ff_layer = tf.keras.layers.Dense(emb_sz, activation='relu')

        self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  if not multiheaded else MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) if not multiheaded else MultiHeadedAttention(emb_sz, False)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.layer_norm3 = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs, context_sequence):
        """
        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        # print("calling self_atten")
        # print("shapes:", np.shape(inputs), np.shape(context_sequence))
        # print(context_sequence)
        # print(inputs)
        masked_attn = self.self_atten(inputs, inputs, inputs)
        # print("got masked_attn")
        masked_attn = masked_attn + inputs
        # print("got masked_attn")
        masked_attn = self.layer_norm1(masked_attn)
        # print("got context_sequence")

        unmasked_attn = self.self_context_atten(context_sequence, context_sequence, masked_attn)
        # print("got unmasked_attn")
        unmasked_attn = unmasked_attn + masked_attn
        unmasked_attn = self.layer_norm2(unmasked_attn)
        # print("got unmasked_attn layer norm 2")
        feed_forward = self.ff_layer(unmasked_attn)
        # print("got feed_forward")
        feed_forward = feed_forward + unmasked_attn
        # print("got feed_forward")
        feed_forward = self.layer_norm3(feed_forward)
        return tf.nn.relu(feed_forward)


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        embeddings = self.embedding(x)
        embeddings = embeddings * tf.sqrt(tf.cast(self.embed_size, dtype=tf.float32))
        embeddings = tf.add(embeddings, self.pos_encoding)
        return embeddings
    