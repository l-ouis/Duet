from music21 import *
from tqdm import tqdm
import numpy as np
import os
from miditok import REMI, TokenizerConfig
from symusic import Score
import pickle
import copy

# This script differs from valid_midi.py in that we use an alternative dataset instead of the Lakh-based dataset.
# Since this dataset does not have the same two-channel structure, we will do some crude assumptions and split 
# melody and harmony off of of the average pitch of the midi file.

# with open('src/data/generated/maestro/tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

config = TokenizerConfig(num_velocities=8)
tokenizer = REMI(config)


raw_data_path = "src/data/maestro_dataset"
valid_data_path = "src/data/valid_midi_maestro"
count = 0
midi_tokens_list = []
break_out = False
for root, dirs, files in os.walk(raw_data_path):
    total_files = len(files)
    for file in files:
        try:
            file_path = os.path.join(root, file)
            score = converter.parse(file_path)

            combined_score = stream.Score()
            for part in score.parts:
                combined_score.insert(0, part)

            # Remove all preceding rests
            for element in combined_score.recurse():
                if isinstance(element, note.Note):
                    break
                elif isinstance(element, note.Rest):
                    combined_score.remove(element)
            # # Quantize the score
            # print("Quantizing score")
            # combined_score.quantize(inPlace=True)
            # combined_score.write('midi', fp='src/temp/quantized_score.mid')
            # print("Quantized score saved")
            midi_score = Score(file_path)
            midi_tokens = tokenizer(midi_score)
            midi_tokens_list.append(midi_tokens)
            count += 1
            print("At file #", count)
        except Exception as e:
            print("Invalid stuff occured when parsing:", file_path)
            print(e)
            continue
with open('src/data/generated/maestro/midi_tokens.pkl', 'wb') as f:
    pickle.dump(midi_tokens_list, f)
with open('src/data/generated/maestro/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Successfully parsed ", count, " files")
