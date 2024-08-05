import pickle

with open('src/data_preprocessing/tokenizers/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print(tokenizer.vocab)
