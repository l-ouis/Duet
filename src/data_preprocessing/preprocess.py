from tqdm import tqdm
import numpy as np
import os
from miditok import Structured, TokenizerConfig
from symusic import Score
import pickle
from concurrent.futures import ProcessPoolExecutor

# This script differs from valid_midi.py in that we use an alternative dataset instead of the Lakh-based dataset.
# Since this dataset does not have the same two-channel structure, we will do some crude assumptions and split 
# melody and harmony off of of the average pitch of the midi file.

# with open('src/data/generated/maestro/tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

config = TokenizerConfig(num_velocities=8)
tokenizer = Structured(config)

data_folder = "src/data_preprocessing"
raw_data_path = "src/data_preprocessing/raw_midi"
tokenized_data_path = "src/data_preprocessing/tokenized_midi"
count = 0
midi_tokens_list = []  
break_out = False

midi_filepaths = [os.path.join(root, file) for root, _, files in os.walk(raw_data_path) for file in files if (file.endswith('.mid') or file.endswith('.midi'))]

def tokenize_midi(file_path: str) -> None:
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        midi_score = Score(file_path)
        midi_score = midi_score.resample(tpq=6, min_dur=1)
        midi_tokens = tokenizer(midi_score)
        with open(os.path.join(tokenized_data_path, file_name + ".pkl"), 'wb') as f:
            pickle.dump(midi_tokens, f)
    except Exception as e:
        print("UB occured when parsing:", file_path)
        print(e)


def collect_tokenized_midi_files() -> list:
    tokenized_files = []
    for root, _, files in os.walk(tokenized_data_path):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    tokenized_file = pickle.load(f)
                    tokenized_file = tokenized_file[0].ids
                    tokenized_files.append(tokenized_file)
    return tokenized_files

def split_tokens_into_pairs(tokenized_files, window_size):
    input_pairs = []
    label_pairs = []
    for file in tokenized_files:
        for i in range(0, len(file) - window_size*2, window_size*2):
            print("did something")
            input_pairs.append(file[i:i+window_size])
            label_pairs.append(file[i+window_size:i+window_size*2])
    return input_pairs, label_pairs





if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        executor.map(tokenize_midi, midi_filepaths)
    with open('src/data_preprocessing/tokenizers/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenized files")
    collected_files = collect_tokenized_midi_files()
    print(len(collected_files))
    input_tokens, label_tokens = split_tokens_into_pairs(collected_files, 64)
    print(len(input_tokens), len(label_tokens))
    with open(data_folder + '/transformer_input_label/input_tokens.pkl', 'wb') as f:
        pickle.dump(input_tokens, f)
    with open(data_folder + '/transformer_input_label/label_tokens.pkl', 'wb') as f:
        pickle.dump(label_tokens, f)
    print("Successfully parsed ", len(midi_filepaths), " files")
