import numpy as np
import re
import tensorflow as tf
from support.load import LoadFiles as lf
import streamlit as st

class TextGenerator():

  def __init__(self, sample_input_text, predict_next_words, with_beam=False):
    self.sample_input_text = sample_input_text
    self.predict_next_words = predict_next_words
    self.with_beam = with_beam

  def load_model_weights(self):
    ## load the model using trained weights

    maxlen = 80
    dropout = 0.2
    rnn_units = 512
    embed_dim = 256
    unicode_vocab_size = 20000

    model = WordRNN(unicode_vocab_size, embed_dim, maxlen, dropout, rnn_units, kernel_reg=None)

    sample_input_batch = np.random.randint(low=1, high=unicode_vocab_size, size=(32,maxlen))
    model(sample_input_batch)

    model.load_weights(lf.trained_model_weights_path)
    return model

  # generate text for next set of words using sample input text 
  def generate_text(self):
    
    # load id_lookup & word_lookup
    word_lookup = lf.word_lookup
    id_lookup = lf.id_lookup

    h_states, c_states = None, None
    sample_input = self.text_standardization(self.sample_input_text)

    next_word = [sample_input]
    result = [sample_input]

    # load model 
    model = self.load_model_weights()
    
    my_bar = st.progress(0)

    # lstm model text generation
    for n in range(self.predict_next_words):
      next_word, h_states, c_states = self.gnerate_next_word(model, next_word, result, word_lookup, id_lookup, h_states=h_states, c_states=c_states, beam=self.with_beam)
      result.extend(next_word)
      my_bar.progress((n+1)/self.predict_next_words)

    result = ' '.join([_ for _ in result if _!='UNK'])
    return result

  ## https://keras.io/examples/generative/text_generation_with_miniature_gpt/

  def gnerate_next_word(self, model, inputs, result, word_lookup, id_lookup, h_states=None, c_states=None, beam=False):

    sample_index = len(result)-1

    # Convert strings to token IDs.
    input_ids = self.get_input_ids(result, word_lookup)

    if beam:
      predicted_logits, h_states, c_states = self.predict_one_step(model, input_ids, h_states, c_states)

      # pick the highest log likelihood ratio among top 3 proposals
      predicted_ids = self.beam_search_decoder(predicted_logits[:,sample_index,:], 3)
      predicted_ids = predicted_ids[-1][0][0]
    
    else:
      predicted_logits, h_states, c_states = self.predict_one_step(model, input_ids, h_states, c_states)
      
      # Convert from token ids to characters
      predicted_ids = self.greedy_seach(predicted_logits[0][sample_index], id_lookup)
    
    predicted_word = id_lookup[predicted_ids]

    # Return the word and model state.
    return [predicted_word], h_states, c_states

  # generate token ids for sample input 
  def get_input_ids(self, sample_input, word_lookup):
    input = np.zeros(80, dtype=np.int64)
    for idx, w in enumerate(sample_input):
      input[idx]=word_lookup.get(w,1)
    return tf.convert_to_tensor(input)

  ##https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

  # beam search
  def beam_search_decoder(self, data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
      all_candidates = list()
      # expand each current candidate
      for i in range(len(sequences)):
        seq, score = sequences[i]
        for j in range(len(row)):
          candidate = [seq + [j], score - row[j]]
          all_candidates.append(candidate)
      # order all candidates by score
      ordered = sorted(all_candidates, key=lambda tup:tup[1])
      # select k best
      sequences = ordered[:k]
    return sequences

  # Get the top preicted index based on logits
  def greedy_seach(self, predicted_logits, id_lookup):
    logits, indices = tf.math.top_k(predicted_logits, k=10, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

  # predict one step/sentence based on the model type
  def predict_one_step(self, model, input, h_states, c_states):

    embd_layer = model.layers[0](input)
    embd_layer = embd_layer.numpy().reshape(1, embd_layer.numpy().shape[0], embd_layer.numpy().shape[1])
    if (h_states is None) & (c_states is None):
        h_states, c_states = model.layers[1].get_initial_state(embd_layer)
    lstm_out, h_states, c_states = model.layers[1](embd_layer, initial_state=[h_states, c_states])
    predicted_logits = model.layers[2](lstm_out)
    return predicted_logits, h_states, c_states

  # converts to lower case, remove non-asci chars and html tags
  def text_standardization(self, text):
    
    HTML = re.compile('<.*?>')
    # initializing punctuations string
    PUNCT = '''!()-[]{};:"\<>/?@#$%^&*_~'''

    text = text.lower()
    text = re.sub(HTML, ' ', text)
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    for ele in text:
      if ele in PUNCT:
          text = text.replace(ele, "")
    return text


## https://keras.io/examples/generative/text_generation_with_miniature_gpt/

## Create two seperate embedding layers: one for tokens and one for token index (positions).

class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

## https://www.tensorflow.org/text/tutorials/text_generation

class WordRNN(tf.keras.Model):

  def __init__(self, vocab_size, embed_dim, maxlen, dropout, rnn_units, kernel_reg=None):
    super().__init__(self)
    self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

    # LSTM layer initialisation
    self.lstm = tf.keras.layers.LSTM(rnn_units, dropout=dropout,
                                    return_sequences=True,
                                    return_state=True)
    
    self.dense = tf.keras.layers.Dense(vocab_size, kernel_regularizer=kernel_reg)

  def call(self, inputs, h_states=None, c_states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding_layer(x, training=training)
    
    if (h_states is None) & (c_states is None):
      h_states, c_states = self.lstm.get_initial_state(x)
    x, h_states, c_states = self.lstm(x, initial_state=[h_states, c_states], training=training)
    x = self.dense(x, training=training)
    
    if return_state:
      return x, h_states, c_states
    else:
      return x