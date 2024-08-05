import numpy as np
from music21 import *
import pickle
from miditok import REMI, TokenizerConfig

# 4000 datapoints for now in these pickled lists
with open('src/data/generated/input_tokens.pkl', 'rb') as f:
    input_tokens_list = pickle.load(f)
with open('src/data/generated/label_tokens.pkl', 'rb') as f:
    label_tokens_list = pickle.load(f)
with open('src/data/generated/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

tokenizer.vocab['Start'] = 258
final_input_ids = []
final_label_ids = []

# Normalize the length of the input and label lists to be the same window length, then output batches into a pickled file
for (input, label) in zip(input_tokens_list, label_tokens_list):
    # The actual token data is in the first element
    input_tokens = input[0]
    label_tokens = label[0]
    input_ids = input_tokens.ids
    input_ids = [input_ids[i] for i in range(len(input_ids)-1) if input_ids[i] != 4 or input_ids[i] != input_ids[i+1]]
    label_ids = label_tokens.ids
    label_ids = [label_ids[i] for i in range(len(label_ids)-1) if label_ids[i] != 4 or label_ids[i] != label_ids[i+1]]

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
            for j in range(-6, 6):
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
with open('src/data/generated/lakh/final_input_ids.pkl', 'wb') as f:
    pickle.dump(final_input_ids, f)
with open('src/data/generated/lakh/final_label_ids.pkl', 'wb') as f:
    pickle.dump(final_label_ids, f)
