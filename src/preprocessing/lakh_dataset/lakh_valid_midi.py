from music21 import *
from tqdm import tqdm
import numpy as np
import os
from miditok import REMI, TokenizerConfig
from symusic import Score
import pickle

# Run this file from the root directory to 

def get_tempo(time, metronome_boundaries):
    for i in metronome_boundaries:
        if time >= i[0] and time < i[1]:
            return i[2]
    return "Invalid time"

config = TokenizerConfig(num_velocities=8)
tokenizer = REMI(config)

raw_data_path = "src/data/raw_lakh"
valid_data_path = "src/data/valid_midi_lakh"
count = 0
input_tokens_list = []
label_tokens_list = []
for root, dirs, files in os.walk(raw_data_path):
    total_files = len(files)
    for file in files:
        if file.endswith(".mid"):
            try:
                file_path = os.path.join(root, file)
                score = converter.parse(file_path)
                if len(score.parts) == 2:
                    print("Valid 2-channel,", file_path, count)

                    # Get each part (right hand, left hand presumably)
                    allparts = score.parts
                    allparts = [part for part in allparts if hasattr(part.first(), 'notes')]
                    part1 = allparts[0]
                    part2 = allparts[1]
                    part1_transpose = []
                    part2_transpose = []

                    # Transpose part1 and part2 to each key in the scale
                    for i in range(-6, 6):
                        transposed_part1 = part1.transpose(i)
                        transposed_part2 = part2.transpose(i)
                        part1_transpose.append(transposed_part1)
                        part2_transpose.append(transposed_part2)
                        tp1_string = 'src/temp/temp_part1_transpose' + str(i) + '.mid'
                        tp2_string = 'src/temp/temp_part2_transpose' + str(i) + '.mid'
                        transposed_part1.write('midi', fp=tp1_string)
                        transposed_part2.write('midi', fp=tp2_string)
                        part1_score = Score(tp1_string)
                        part2_score = Score(tp2_string)
                        input_tokens = tokenizer(part1_score)
                        label_tokens = tokenizer(part2_score)
                        input_tokens_list.append(input_tokens)
                        label_tokens_list.append(label_tokens)
                        print("transposition done for ", file_path, i)
                    # input_midi = tokenizer(input_tokens)
                    # label_midi = tokenizer(label_tokens)
                    # input_midi.dump_midi("out_input.mid")
                    # label_midi.dump_midi("out_label.mid")
                    count += 1
                            
                    # Should put these in valid_midi
                    # When running the count script only, there are 4653 valid files with 2 
                    # Maybe check for midis that also have "piano 1" "piano 2"?
            except Exception as e:
                print("Invalid behavior occured when parsing:", file_path)
                print(e)
            continue
with open('src/data/generated/lakh/input_tokens.pkl', 'wb') as f:
    pickle.dump(input_tokens_list, f)
with open('src/data/generated/lakh/label_tokens.pkl', 'wb') as f:
    pickle.dump(label_tokens_list, f)
with open('src/data/generated/lakh/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Successfully parsed ", count, " files")
