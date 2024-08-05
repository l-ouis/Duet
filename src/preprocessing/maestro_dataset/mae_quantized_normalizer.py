import numpy as np
from music21 import *
import pickle
from miditok import REMI, TokenizerConfig


# Edit these print statements for the dataset you're using.
with open('src/data/generated/maestro/midi_tokens.pkl', 'rb') as f:
    midi_tokens_list = pickle.load(f)
with open('src/data/generated/maestro/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

tokenizer.vocab['Start'] = 272
# print("Length of pickled input and label lists:")
# print(len(midi_tokens_list))
# print(tokenizer.vocab)
final_input_ids = []
final_label_ids = []
rev_voc = {v: k for k, v in tokenizer.vocab.items()}


for tokens in midi_tokens_list:
    token_ids = tokens[0].ids
    # The actual token data is in the first element
    tokens_no_bar = [token_ids[i] for i in range(len(token_ids)-1) if token_ids[i] != 4 or token_ids[i] != token_ids[i+1]]
    
    current_input_ids = tokens_no_bar
    current_label_ids = tokens_no_bar
    
    # Middle C is Pitch_60 : 44
    # Pitches go from 5 to 92
    filtered_input_ids = []
    i = 0
    while i < len(current_input_ids):
        if 5 <= current_input_ids[i] <= 43:
            i += 3  # Skip the current and the next 2 elements
        else:
            filtered_input_ids.append(current_input_ids[i])
            i += 1
    input_ids = filtered_input_ids
    filtered_label_ids = []
    i = 0
    while i < len(current_label_ids):
        if 44 <= current_label_ids[i] <= 92:
            i += 3  # Skip the current and the next 2 elements
        else:
            filtered_label_ids.append(current_label_ids[i])
            i += 1
    label_ids = filtered_label_ids

    

    min_length = min(len(input_ids), len(label_ids))
    batches = min_length // 256
    if min_length > 256:
        # Split the input and label ids into batches of 256 until they can't be split anymore
        input_batches = [input_ids[i*256:(i+1)*256] for i in range(batches)]
        label_batches = [label_ids[i*256:(i+1)*256] for i in range(batches)]
        final_input_batches = []
        final_label_batches = []

        # If the id is between 5 and 92 it is a pitch, so transpose it from range(-6, 6).
        for (input_sequence, label_sequence) in zip(input_batches, label_batches):
            if any(id in range(197, 258) for id in input_sequence) or any(id in range(197, 258) for id in label_sequence):
                continue
            input_sequence = [258] + input_sequence
            label_sequence = [258] + label_sequence
            input_holder = input_sequence
            label_holder = label_sequence
            for j in range(-6, 6):
                input_sequence = input_holder
                label_sequence = label_holder
                for i in range(len(input_sequence)):
                    if 11 <= input_sequence[i] <= 87:
                        input_sequence[i] = input_sequence[i] + j
                final_input_batches.append(input_sequence)
                for i in range(len(label_sequence)):
                    if 11 <= label_sequence[i] <= 87:
                        label_sequence[i] = label_sequence[i] + j
                final_label_batches.append(label_sequence)

        # Each element in these batches will be a single data point.
        # Append these to the final lists
        final_input_ids.extend(final_input_batches)
        final_label_ids.extend(final_label_batches)


# Save final_input_ids and final_label_ids to a pickled file
with open('src/data/generated/maestro/final_input_ids.pkl', 'wb') as f:
    pickle.dump(final_input_ids, f)
with open('src/data/generated/maestro/final_label_ids.pkl', 'wb') as f:
    pickle.dump(final_label_ids, f)
