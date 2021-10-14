import os
import datetime

import constants
from models.utils import *


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs, **kwargs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


class TransformerModel:
    def __init__(self, vocab, num_layers, units, d_model, num_heads, dropout, model_name):
        if not os.path.exists(constants.TRAINED_MODELS):
            os.mkdir(constants.TRAINED_MODELS)

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.model_name = model_name

    def encoder_layer(self, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        attention = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention")({
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': padding_mask
            })
        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def encoder(self, name="encoder"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        embeddings = tf.keras.layers.Embedding(self.vocab_size + 1, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size + 1, self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)
        for i in range(int(self.num_layers)):
            outputs = self.encoder_layer(name="encoder_layer_{}".format(i),)([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def decoder_layer(self, name="decoder_layer"):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        attention1 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_1")(inputs={
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': look_ahead_mask
            })
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_2")(inputs={
                'query': attention1,
                'key': enc_outputs,
                'value': enc_outputs,
                'mask': padding_mask
        })
        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def decoder(self, name='decoder'):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        embeddings = tf.keras.layers.Embedding(self.vocab_size + 1, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size + 1, self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(int(self.num_layers)):
            outputs = self.decoder_layer(
                name='decoder_layer_{}'.format(i),
            )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def transformer(self, name="transformer"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(
            create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder()(inputs=[inputs, enc_padding_mask])

        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=self.vocab_size, name="outputs")(dec_outputs)

        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

    def _build(self):
        # clear backend
        tf.keras.backend.clear_session()

        self.learning_rate = CustomSchedule(self.d_model)

        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        with constants.strategy.scope():
            self.model = self.transformer()

            self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=[accuracy])

        self.model.summary()

    def train(self, dataset_train, dataset_val):
        self._build()
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        self.model.fit(dataset_train, validation_data=dataset_val, epochs=constants.EPOCHS,
                       callbacks=[tensorboard_callback])

        self.model.save(os.path.join(constants.TRAINED_MODELS, self.model_name))

    def evaluate(self, sentence):
        sentence = tf.expand_dims(sentence, axis=0)

        output = tf.expand_dims([self.vocab[constants.START]], axis=0)

        end = self.vocab[constants.END]

        for i in range(constants.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, end):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, sentence):
        self.model.load(os.path.join(constants.TRAINED_MODELS, self.model_name))
        prediction = self.evaluate(sentence)

        predicted_sentence = decode(prediction, self.vocab)

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence
