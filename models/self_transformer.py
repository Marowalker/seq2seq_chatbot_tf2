import os
import numpy as np
import constants
from models.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
import tensorflow_text as tft


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

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'depth': self.depth,
            # 'query_dense': self.query_dense.numpy(),
            # 'key_dense': self.key_dense.numpy(),
            # 'value_dense': self.value_dense.numpy(),
            # 'dense': self.dense
        }
        return config


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

    def get_config(self):
        config = {
            'pos_encoding': self.pos_encoding
        }
        return config


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


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    for i in range(int(num_layers)):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(int(num_layers)):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
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

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


class TransformerModel:
    def __init__(self, tokenizer, num_layers, units, d_model, num_heads, dropout, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_counts) + 1
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.model_path = model_path
        self.model = None

        self._build()

    def _build(self):
        # clear backend
        tf.keras.backend.clear_session()

        self.learning_rate = CustomSchedule(self.d_model)

        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        with constants.strategy.scope():
            self.model = transformer(self.vocab_size, self.num_layers, self.units, self.d_model, self.num_heads,
                                     self.dropout)

            self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=[accuracy])

        self.model.summary()

    def train(self, dataset_train, dataset_val):
        # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path + 'ckpt.{epoch:02d}.hdf5',
            monitor='val_accuracy',
            mode='auto',
            save_freq='epoch',
            save_best_only=True,
            save_weights_only=True
        )

        if os.listdir(self.model_path):
            print("Load model from last checkpoint...\n")
            checkpoint_file, init_epoch = get_checkpoint(self.model_path)
            self.model.load_weights(self.model_path + checkpoint_file)
        else:
            init_epoch = 0

        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

        self.model.fit(dataset_train, validation_data=dataset_val, epochs=constants.EPOCHS, initial_epoch=init_epoch,
                       callbacks=[model_checkpoint_callback])

    def evaluate(self, sentence):
        sentence = self.tokenizer.texts_to_sequences(sentence)[0]
        sentence = tf.expand_dims(sentence, axis=0)

        output = self.tokenizer.texts_to_sequences([constants.START])[0]
        output = tf.expand_dims(output, axis=0)

        end = self.tokenizer.texts_to_sequences([constants.END])[0]

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

    def evaluate_coherence(self, inputs):
        checkpoint_file, init_epoch = get_checkpoint(self.model_path)
        self.model.load_weights(self.model_path + checkpoint_file)

        predictions = []

        for sentence in inputs:
            prediction = self.evaluate(sentence)
            predictions.append(prediction.numpy().tolist()[1:])

        # print(predictions)
        predictions = pad_sequences(predictions, maxlen=constants.MAX_LENGTH, padding='post')
        answers = self.tokenizer.texts_to_sequences(inputs)
        answers = pad_sequences(answers, maxlen=constants.MAX_LENGTH, padding='post')
        print(1 - np.average(cosine_similarity(answers, predictions)))
        print('\n')

    def evaluate_bleu(self, inputs, outputs):
        checkpoint_file, init_epoch = get_checkpoint(self.model_path)
        self.model.load_weights(self.model_path + checkpoint_file)

        predictions = []

        for sentence in inputs:
            prediction = self.evaluate(sentence)
            answer = self.tokenizer.sequences_to_texts([prediction.numpy().tolist()[1:]])[0]
            predictions.append(answer.split())

        answers = [i.split() for i in outputs]
        bleu_1 = corpus_bleu(answers, predictions, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(answers, predictions, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(answers, predictions, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(answers, predictions)
        print("BLEU-1 score: ", bleu_1)
        print("BLEU-2 score: ", bleu_2)
        print("BLEU-3 score: ", bleu_3)
        print("BLEU-4 score: ", bleu_4)
        print('\n')

    def evaluate_rouge(self, inputs, outputs):
        checkpoint_file, init_epoch = get_checkpoint(self.model_path)
        self.model.load_weights(self.model_path + checkpoint_file)

        predictions = []

        for sentence in inputs:
            prediction = self.evaluate(sentence)
            answer = self.tokenizer.sequences_to_texts([prediction.numpy().tolist()[1:]])[0]
            predictions.append(answer.split())

        answers = [i.split() for i in outputs]

        predictions = tf.ragged.constant(predictions)
        answers = tf.ragged.constant(answers)
        result = tft.metrics.rouge_l(predictions, answers)
        print('F-Measure: %s' % np.average(result.f_measure.numpy()))
        print('P-Measure: %s' % np.average(result.p_measure.numpy()))
        print('R-Measure: %s' % np.average(result.r_measure.numpy()))
        print('\n')

    def predict(self, sentence):
        checkpoint_file, init_epoch = get_checkpoint(self.model_path)
        self.model.load_weights(self.model_path + checkpoint_file)

        prediction = self.evaluate(sentence)

        # print(prediction.numpy())

        # predicted_sentence = decode(prediction.numpy(), self.vocab)
        # print(prediction.to_list())
        predicted_sentence = self.tokenizer.sequences_to_texts([prediction.numpy().tolist()[1:]])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence
