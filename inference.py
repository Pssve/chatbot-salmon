# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 06:16:54 2021

@author: Dean
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.preprocessing.sequence import pad_sequences
from chatbot_data import *
from encoder_decoder import *

enc_model = Model([enc_inp], enc_state)

#this is the decoder model
decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = dec_lstm(dec_embed, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

dec_model = Model([dec_inp]+decoder_states_inputs, [decoder_outputs]+decoder_states)

print("################################")
print("###      start chatting      ###")
print("################################")

prepro1=""
while prepro1 != 'q':
    prepro1 = input("you: ")
    #prepro1 = Hello
    
    prepro1 = clean_text(prepro1)
    #prepro1 = hello
    
    prepro1 = [prepro1]
    #prepro1 = [hello]
    
    txt = []
    for x in prepro1:
        lst = []
        for y in x.split:
            try:
                lst.append(vocab[y])
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)
            
    txt = pad_sequences(txt, 13, padding='post')
    
    stat = enc_model.predict( txt )
    
    empty_target_seq = np.zeros( ( 1, 1 ) )
    
    empty_target_seq[0, 0] = vocab['<SOS>']
    
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition:
        
        dec_outputs, h, c = dec_model.predict([empty_target_seq]+stat)
        decoder_concat_input = dense(dec_outputs)
        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        sampled_word = inv_vocab[sampled_word_index] + ' '
        
        
        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word
        
        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
            stop_condition = True
            
        empty_target_seq = np.zeros( (1, 1 ) )
        empty_target_seq[0, 0] = sampled_word_index
        stat = [h, c]
    
    print("Chatbot attention: ", decoded_translation)
    print("==============================================")
    