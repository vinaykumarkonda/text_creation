import pandas as pd
import os

file_path = '../model_training/production_data/'#os.getcwd()+


class LoadFiles():

    # load trained model weights
    trained_model_weights_path = file_path+'scifi_lstm_model_with_full_data_unicode_20k_vocab_size.hdf5'

    # load word_lookup
    word_lookup = pd.read_pickle(file_path+'scifi_unicode_vocab_dict.pkl')

    # load id_lookup
    id_lookup = pd.read_pickle(file_path+'scifi_unicode_id_lookup.pkl')
