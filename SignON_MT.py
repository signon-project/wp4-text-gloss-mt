# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:56:11 2023

@author: SNT
"""

import sentencepiece as spm
import numpy as np
import pickle
import tensorflow as tf 
tf.config.set_visible_devices([], 'GPU')
from transformers.models.mbart.modeling_tf_mbart import (TFMBartEncoder, MBartConfig, 
                                                         TFSharedEmbeddings, TFMBartDecoder)
import os


# ALLOWED LANGUAGES 
ALLOWED_SPOKEN_LAN = ['es_XX', 'en_XX', 'de_DE', 'nl_XX']
ALLOWED_SIGN_LAN = ['LSE', 'BSL', 'DGS', 'NGT']



class SignON_MT:
    def __init__(self):
        """
        Construtor for the class SignON_MT. Example:
            model = SignON_MT()
            spoken_input = 'tiefer luftdruck bestimmt in den nÃ¤chsten tagen unser wetter'
            gloss_input = 'DRUCK TIEF KOMMEN'

            encoded_sentence = model.encode(gloss_input.lower(), 'DGS')
            decoded_sentence = model.decode(encoded_sentence, 'de_DE')

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.tokenizer = spm.SentencePieceProcessor(model_file='./models/spm_mbart.model')
            
        # LOADING APPROACH 1 MAPPERS AND MODEL WEIGHTS
        self.model_name = "model"
    
        with open(os.path.join('mappers', self.model_name+'.mappers'), 'rb') as f:
            data_mappers = pickle.load(f)             
        with open(os.path.join('models', self.model_name+'.weights'), 'rb') as f:
            model_weights = pickle.load(f)             
                

        # APPROACH 1 - MAPPERS - WEIGHTS
        self.spoken_id_mapper = dict(data_mappers['id_mapper'])
        self.spoken_reverse_id_mapper = dict([(a[1], a[0]) for a in data_mappers['id_mapper']])
        self.spoken_special_tokens = dict(data_mappers['special_tokens'])
        self.spoken_mbart_table, self.spoken_mbart_encoder, self.spoken_mbart_decoder = self.load_mbart(vocab_size = len(data_mappers['id_mapper']))
        
        # The branch to process glosses are the same as spoken text
        self.gloss_special_tokens = self.spoken_special_tokens
        self.gloss_id_mapper =  self.spoken_id_mapper
        self.gloss_reverse_id = self.spoken_reverse_id_mapper
        self.gloss_mbart_table = self.spoken_mbart_table 
        self.gloss_mbart_encoder = self.spoken_mbart_encoder
        self.gloss_mbart_decoder = self.spoken_mbart_decoder
        
        # Loading weights
        assert all([True if k in ['token_table', 'encoder', 'decoder'] else False for k in model_weights.keys() ]) , 'Weight format incorrect'
        self.spoken_mbart_table.set_weights(model_weights['token_table'])
        self.spoken_mbart_encoder.set_weights(model_weights['encoder'])
        self.spoken_mbart_decoder.set_weights(model_weights['decoder'])
        
        # Creating tokenization and mapping functions
        self.spoken_map_lambda = lambda x : self.spoken_id_mapper.get(x) if x in self.spoken_id_mapper.keys() else self.spoken_special_tokens['<unk>']
        def spoken_tokenize(input_text):
            tokenized = list(map(self.spoken_map_lambda, self.tokenizer.Encode(input_text)))
            return tokenized
        
        # Creating detokenization and reversed-mapping functions
        self.spoken_reverse_map_lambda = lambda x : self.spoken_reverse_id_mapper.get(x) if x in self.spoken_reverse_id_mapper.keys() else self.spoken_special_tokens['<unk>']
        def spoken_detokenize(input_seq):
            input_seq_f = [int(i) for i in input_seq if i not in self.spoken_special_tokens.values()]  
            input_seq_m = list(map(self.spoken_reverse_map_lambda,input_seq_f))
            input_seq_m = list(map(int,input_seq_m))
            return self.tokenizer.Detokenize(input_seq_m)            
        
        self.spoken_tokenize = spoken_tokenize
        self.gloss_map_lambda = self.spoken_map_lambda
        self.gloss_tokenize = self.spoken_tokenize
        self.spoken_detokenize = spoken_detokenize    
        self.gloss_detokenize = self.spoken_detokenize
            
            
    def load_mbart(self, vocab_size: int = None, in_cpu: bool  = True) -> (TFSharedEmbeddings, TFMBartEncoder, TFMBartDecoder):
        """
        This function loads mBART model with an specific vocabulary size.

        Parameters
        ----------
        vocab_size : int, optional
            Vocabulary size for mBART. The default is None.
        in_cpu : bool, optional
            If true embedding table is loaded in CPU. The default is True.

        Returns
        -------
        token_embeddings : TFSharedEmbeddings
            mBART embedding table.
        encoder : TFMBartEncoder
            mBART Encoder.
        decoder : TFMBartDecoder
            mBART Decoder.

        """
        
        # LOADING MBART CONFIG OBJECT
        with open(os.path.join('./models/','config_architecture.dict'), 'rb') as f:
            config_dict = pickle.load(f)
            
        vocab_size_c = config_dict['vocab_size'] if 'vocab_size' in config_dict else 50265
        max_position_embeddings= config_dict['max_position_embeddings'] if 'max_position_embeddings' in config_dict else 1024 
        encoder_layers= config_dict['encoder_layers'] if 'encoder_layers' in config_dict else 12
        encoder_ffn_dim= config_dict['encoder_ffn_dim'] if 'encoder_ffn_dim' in config_dict else 4096
        encoder_attention_heads= config_dict['encoder_attention_heads'] if 'encoder_attention_heads' in config_dict else 16
        decoder_layers= config_dict['decoder_layers'] if 'decoder_layers' in config_dict else 12
        decoder_ffn_dim= config_dict['decoder_ffn_dim'] if 'decoder_ffn_dim' in config_dict else 4096
        decoder_attention_heads= config_dict['decoder_attention_heads'] if 'decoder_attention_heads' in config_dict else 16
        encoder_layerdrop= config_dict['encoder_layerdrop'] if 'encoder_layerdrop' in config_dict else 0.0
        decoder_layerdrop= config_dict['decoder_layerdrop'] if 'decoder_layerdrop' in config_dict else 0.0
        use_cache= config_dict['use_cache'] if 'use_cache' in config_dict else True
        is_encoder_decoder= config_dict['is_encoder_decoder'] if 'is_encoder_decoder' in config_dict else True
        activation_function= config_dict['activation_function'] if 'activation_function' in config_dict else "gelu"
        d_model= config_dict['d_model'] if 'd_model' in config_dict else 1024
        dropout= config_dict['dropout'] if 'dropout' in config_dict else 0.1
        attention_dropout= config_dict['attention_dropout'] if 'attention_dropout' in config_dict else 0.0
        activation_dropout= config_dict['activation_dropout'] if 'activation_dropout' in config_dict else 0.0
        init_std= config_dict['init_std'] if 'init_std' in config_dict else 0.02
        classifier_dropout= config_dict['classifier_dropout'] if 'classifier_dropout' in config_dict else 0.0
        scale_embedding= config_dict['scale_embedding'] if 'scale_embedding' in config_dict else False
        gradient_checkpointing= config_dict['gradient_checkpointing'] if 'gradient_checkpointing' in config_dict else False
        pad_token_id= config_dict['pad_token_id'] if 'pad_token_id' in config_dict else 1
        bos_token_id= config_dict['bos_token_id'] if 'bos_token_id' in config_dict else 0
        eos_token_id= config_dict['eos_token_id'] if 'eos_token_id' in config_dict else 2
        forced_eos_token_id= config_dict['forced_eos_token_id'] if 'forced_eos_token_id' in config_dict else 2
    
        mbartConfig_dict = MBartConfig(vocab_size=vocab_size_c, max_position_embeddings=max_position_embeddings,
                                        encoder_layers=encoder_layers, encoder_ffn_dim=encoder_ffn_dim,
                                        encoder_attention_heads=encoder_attention_heads, decoder_layers=decoder_layers,
                                        decoder_ffn_dim=decoder_ffn_dim, decoder_attention_heads=decoder_attention_heads,
                                        encoder_layerdrop=encoder_layerdrop, decoder_layerdrop=decoder_layerdrop,
                                        use_cache=use_cache, is_encoder_decoder=is_encoder_decoder,
                                        activation_function=activation_function, d_model=d_model,
                                        dropout=dropout, attention_dropout=attention_dropout,
                                        activation_dropout=activation_dropout, init_std=init_std,
                                        classifier_dropout=classifier_dropout, scale_embedding=scale_embedding,
                                        gradient_checkpointing=gradient_checkpointing, pad_token_id=pad_token_id,
                                        bos_token_id=bos_token_id, eos_token_id=eos_token_id, forced_eos_token_id=forced_eos_token_id)        
                
        if vocab_size != None:
            mbartConfig_dict.vocab_size = vocab_size
            
        # CREATING MODEL
        if in_cpu:
            with tf.device('cpu:0'):
                token_embeddings = TFSharedEmbeddings(mbartConfig_dict.vocab_size, mbartConfig_dict.d_model)    
        else:
            token_embeddings = TFSharedEmbeddings(mbartConfig_dict.vocab_size, mbartConfig_dict.d_model)    
        encoder = TFMBartEncoder(mbartConfig_dict)
        decoder = TFMBartDecoder(mbartConfig_dict)
        
        # INITIALIZING MODEL
        w_embs = token_embeddings([[0,1,2,3]])
        encoder_outputs = encoder(None, inputs_embeds = w_embs,
                              attention_mask=None)[0]
        decoder(None, inputs_embeds=w_embs, encoder_attention_mask=None,
                encoder_hidden_states=encoder_outputs, 
                return_dict = True,
                past_key_values = None)   
        
        return token_embeddings, encoder, decoder
    
    def encode(self, input_text : str, src_lan : str) -> np.array:
        """
        Endoces an input text to interl representation

        Parameters
        ----------
        input_text : str
            Spoken or gloss input text.
        src_lan : str
            String to specify the source language (see ALLOWED_SPOKEN_LAN and 
                                                   ALLOWED_SIGN_LAN).

        Returns
        -------
        np.array
            Array with the encoded information.

        """
        
        assert src_lan in ALLOWED_SPOKEN_LAN+ALLOWED_SIGN_LAN, 'Source language not recognized' 
        modality_type = 0 if src_lan in ALLOWED_SIGN_LAN else 1 # 0 = Glosses / 1 = Spoken
        # Processing depending on the modality
        if modality_type == 1:
            enc = self.spoken_mbart_encoder
            table = self.spoken_mbart_table                
            input_ids = [self.spoken_special_tokens['<s>'], self.spoken_special_tokens['<'+src_lan+'>']]+ \
                self.spoken_tokenize(input_text) + [self.spoken_special_tokens['</s>']]
            pad_val =  self.spoken_special_tokens['<pad>']
        else: 
            enc = self.gloss_mbart_encoder
            table = self.gloss_mbart_table                
            input_ids = [self.gloss_special_tokens['<s>'], self.gloss_special_tokens['<'+src_lan+'>']]+ \
                self.gloss_tokenize(input_text) + [self.gloss_special_tokens['</s>']]
            pad_val =  self.gloss_special_tokens['<pad>']
            
        input_ids = np.expand_dims(np.array(input_ids),0)
        input_attention = (input_ids != pad_val).astype('int')    

        input_embs = table(input_ids) * 32.0
        enc_hidden_states = enc(input_ids = None, training = False,
                inputs_embeds = input_embs, attention_mask=input_attention)['last_hidden_state']
        return enc_hidden_states.numpy()
            
    def decode(self, interl_seq : np.array, tgt_lan : str, input_attention : np.array = None) -> str:
        """
        Decodes from interl representation to text

        Parameters
        ----------
        interl_seq : np.array
            Array with the encoded representation of a source sentence.
        tgt_lan : str
            String to specify the target language (see ALLOWED_SPOKEN_LAN and 
                                                   ALLOWED_SIGN_LAN).
        input_attention : np.array, optional
            Array with attention mask. The default is None.

        Returns
        -------
        str
            Decoded sentence.

        """
        
        assert tgt_lan in ALLOWED_SPOKEN_LAN+ALLOWED_SIGN_LAN, 'Target language not recognized' 
        modality_type = 0 if tgt_lan in ALLOWED_SIGN_LAN else 1 # 0 = Glosses / 1 = Spoken
        if modality_type == 1:
            dec = self.spoken_mbart_decoder
            table = self.spoken_mbart_table                
            bos_token_id = self.spoken_special_tokens['<s>']
            target_lan_id = self.spoken_special_tokens['<'+tgt_lan+'>']
            eos_token_id =  self.spoken_special_tokens['</s>']
            detokenize_fn = self.spoken_detokenize
        else: 
            dec = self.gloss_mbart_decoder
            table = self.gloss_mbart_table                
            bos_token_id = self.gloss_special_tokens['<s>']
            target_lan_id = self.gloss_special_tokens['<'+tgt_lan+'>']
            eos_token_id =  self.gloss_special_tokens['</s>']
            detokenize_fn = self.gloss_detokenize

        if input_attention is None:
            input_attention = np.ones((1, interl_seq.shape[1]), dtype = 'int')     
            
        batch_size = 1      
        decoder_ids = tf.expand_dims(tf.stack([bos_token_id, target_lan_id], axis = -1), axis=0)

        # Generation loop
        cur_len = 2
        padding_mask = tf.ones((batch_size, 1), dtype = tf.int32)
        while cur_len < 200:
            decoder_token_embs = table(decoder_ids)*32.0       
            outputs = dec(None,
                    inputs_embeds=decoder_token_embs,
                    encoder_attention_mask=input_attention,
                    encoder_hidden_states=interl_seq, 
                    return_dict = True,
                    past_key_values = None,
                    training = False
                    )    
            
            logits = table(outputs[0], mode = 'linear')
            next_token = tf.expand_dims(tf.gather(tf.argmax(logits, axis = -1, output_type = tf.int32), cur_len-1, axis = -1), axis = -1)
            padding_mask = tf.cast((next_token != eos_token_id), tf.int32)*padding_mask
            decoder_ids = tf.concat([decoder_ids, next_token*padding_mask], axis = -1)
            cur_len += 1
            if sum(padding_mask) == 0: break 
        
        return detokenize_fn(decoder_ids.numpy().tolist()[0])

#%%
'''
EXAMPLE:
'''
model = SignON_MT()
spoken_input = 'tiefer luftdruck bestimmt in den nÃ¤chsten tagen unser wetter'
gloss_input = 'DRUCK TIEF KOMMEN'

encoded_sentence = model.encode(gloss_input.lower(), 'DGS')
decoded_sentence = model.decode(encoded_sentence, 'de_DE')
print(decoded_sentence)

# Output:
#     der tiefdruck zieht weiter nach norddeutschland



