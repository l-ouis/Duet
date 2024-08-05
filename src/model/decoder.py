import tensorflow as tf
import numpy as np

from model.transformer import TransformerBlock, PositionalEncoding


class TransformerDecoder(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Define feed forward layer(s) to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(hidden_size, activation = 'relu')

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(hidden_size)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size)

    def call(self, encoded_images, captions):
        img_embeds = self.image_embedding(tf.expand_dims(encoded_images, 1))
        # print("got img_embeds")
        capt_embeds = self.encoding(captions)
        # print("got capt_embeds")
        # print(np.shape(img_embeds), np.shape(capt_embeds))
        decode_out = self.decoder(capt_embeds, img_embeds)
        # print("got decode_out")
        logits = self.classifier(decode_out)
        # print("got logits")
        return logits
