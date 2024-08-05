from music21 import *
from tqdm import tqdm
import numpy as np
import os
from miditok import REMI, TokenizerConfig
from symusic import Score
import pickle

# Converts a single MIDI to a token sequence

# Specify tokenizer here
with open('src/data_preprocessing/tokenizers/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Specify file here
file_path = "src/testing/test_midis/untitled.mid"
count = 0
final_input_ids = []
input_score = Score(file_path)
input_tokens = tokenizer(input_score)[0]
input_ids = input_tokens.ids
input_ids = [input_ids[i] for i in range(len(input_ids)-1) if input_ids[i] != 4 or input_ids[i] != input_ids[i+1]]

# Dupe input IDs until it meets correct length
while len(input_ids) < 64:
    input_ids.extend(input_ids)

input_ids = input_ids[:64]
input_ids = [1] + input_ids

test_input_ids = [input_ids]

# Save final_input_ids and final_label_ids to a pickled file
with open('src/testing/test_ids/test_input_ids.pkl', 'wb') as f:
    pickle.dump(test_input_ids, f)

print("Successfully parsed test file")

