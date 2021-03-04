# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 05:57:03 2021

@author: Dean
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from chatbot_data import *

enc_inp = Input(shape=(13, ))
dec_inp = Input(shape=(13, ))


VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE+1, output_dim=50,
                   input_length=13,
                   trainable=True
                   )


enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_state = [h, c]


dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_op, _, _ = enc_lstm(enc_embed)

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(dec_op)

model = Model([enc_inp, dec_inp], dense_op)

model.compile(loss='categorical_crossentropy',
              metric=['acc'],
              optimizer='adam')

model.fit([encoder_inp, decoder_inp],
          decoder_final_output,
          epochs=40)

 