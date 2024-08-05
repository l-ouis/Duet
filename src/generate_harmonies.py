import tensorflow as tf
import numpy as np
import numpy as np
from music21 import *
from music21 import stream
import pickle
from miditok import REMI, TokenizerConfig
from model.model import AccompanimentModel, accuracy_function, loss_function

# TODO: implement this in main.py instead of here.


with open('src/testing/test_ids/test_input_ids.pkl', 'rb') as f:
    input_tokens_list = pickle.load(f)
with open('src/data_preprocessing/tokenizers/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


print("length of vocab:")
print(len(tokenizer.vocab))
print(tokenizer.vocab)

# Test loop
for i in range(1):
    input_tokens = np.array(input_tokens_list[i])

# Uncomment to check if the input and labels line up
# input_label_lines_up()

# Load the model
model = tf.keras.models.load_model(
    'src/saved_models/model_duet.keras',
    custom_objects={
        'AccompanmientModel': AccompanimentModel,
    },
    )

def gen_next_notes(model, image_embedding, wordToIds, padID, temp, window_length):
    # idsToWords = {id: word for word, id in wordToIds.items()}
    # unk_token = wordToIds['<unk>']
    # caption_so_far = [wordToIds['<start>']]
    unk_token = 0
    notes_so_far = [1]
    while len(notes_so_far) < window_length:
        notes_input = np.array([notes_so_far + ((window_length - len(notes_so_far)) * [padID])])
        logits = model(np.expand_dims(image_embedding, 0), notes_input[:,:-1])
        logits = logits[0][len(notes_so_far) - 1]
        probs = tf.nn.softmax(logits / temp).numpy()
        next_token = unk_token
        attempts = 0
        while (next_token == unk_token) and attempts < 10:
            next_token = np.random.choice(len(probs), p=probs)
            attempts += 1
        notes_so_far.append(next_token)
    return notes_so_far

for i in range(10):
    for temp_offset in range(5, 10):
        temperature = 1 + (temp_offset * 0.1)

        output = gen_next_notes(model, input_tokens, tokenizer.vocab, 0, temperature, 257)

        # Remove all instances of 258 from the output
        output = [token for token in output if token != 258]
        inp_tokens = input_tokens[1:].tolist()
        # Print the output
        melody = tokenizer.decode([inp_tokens])
        melody.dump_midi("src/testing/test_outputs/test_input.mid")
        input_midi = tokenizer.decode([output])
        input_midi.dump_midi("src/testing/test_outputs/test_output.mid")


        # Load the MIDI files
        input_stream = converter.parse("src/testing/test_outputs/test_input.mid")
        output_stream = converter.parse("src/testing/test_outputs/test_output.mid")

        # Combine the two streams into one
        combined_stream = stream.Score()
        for part in input_stream.parts:
            combined_stream.insert(0, part)
        for part in output_stream.parts:
            combined_stream.insert(0, part)

        # Save the combined MIDI file
        combined_stream.write('midi', fp=f'src/test_outputs/combined_test_{i}_{temp_offset}.mid')
        print(f"Combined MIDI saved as 'src/test_outputs/combined_test_{i}_{temp_offset}.mid'")


def input_label_lines_up():
    with open('src/data/generated/maestro/final_input_ids.pkl', 'rb') as f:
        input_ids_list = pickle.load(f)
    with open('src/data/generated/maestro/final_label_ids.pkl', 'rb') as f:
        label_ids_list = pickle.load(f)
    input_tokens = np.array(input_ids_list[61])
    label_tokens = np.array(label_ids_list[61])[1:].tolist()
    print(input_tokens)
    print(label_tokens)
    testl = []
    rev_voc = {v: k for k, v in tokenizer.vocab.items()}

    for i in input_tokens:
        testl.append(rev_voc[i])
    print(testl)
    melody = tokenizer.decode([input_tokens])
    harmon = tokenizer.decode([label_tokens])

    melody.dump_midi("src/testing/test_outputs/test_input.mid")
    harmon.dump_midi("src/testing/test_outputs/test_output.mid")


    # Load the MIDI files
    input_stream = converter.parse("src/testing/test_outputs/test_input.mid")
    output_stream = converter.parse("src/testing/test_outputs/test_output.mid")

    # Combine the two streams into one
    combined_stream = stream.Score()
    for part in input_stream.parts:
        combined_stream.insert(0, part)
    for part in output_stream.parts:
        combined_stream.insert(0, part)

    combined_stream.write('midi', fp='src/testing/test_outputs/combined_test.mid')
    print("Combined MIDI saved as 'combined_test.mid'")
    exit()
