import numpy as np
import tensorflow as tf
import pickle
from miditok import REMI, TokenizerConfig
import symusic

from model.model import AccompanimentModel, accuracy_function, loss_function
from model.decoder import TransformerDecoder
from model.transformer import AttentionHead
import model.transformer

import os
import argparse
import tensorflow as tf
from typing import Optional
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Check the number of available GPUs
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(f"Number of GPUs available: {num_gpus}")


tf.compat.v1.enable_eager_execution() # Required to not break tf.where for mask

config = TokenizerConfig(num_velocities=16)
tokenizer = REMI(config)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type',           required=True,              choices=['rnn', 'transformer'],     help='Type of model to train')
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both'],  help='Task to run')
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=50,     help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    parser.add_argument('--chkpt_path',     default='src/saved_models/model_duet.h5',                 help='where the model checkpoint is')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def main(args):

    ##############################################################################
    ## Data Loading: These are lists of lists of ID integers. Need to fit them through an embedding layer.
    with open('src/data_preprocessing/transformer_input_label/input_tokens.pkl', 'rb') as f:
        input_tokens = pickle.load(f)
    with open('src/data_preprocessing/transformer_input_label/label_tokens.pkl', 'rb') as f:
        label_tokens = pickle.load(f)
    with open('src/data_preprocessing/tokenizers/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Data loaded!")

    l = len(input_tokens)

    vocab = tokenizer.vocab
    inv_vocab = {v: k for k, v in vocab.items()}
    
    word2idx = vocab
    idx2word = inv_vocab
    input = np.array(input_tokens)
    label = np.array(label_tokens)

    # Shuffling the input and label arrays with the same indices to maintain alignment
    indices = np.arange(len(input))
    np.random.shuffle(indices)
    input = input[indices]
    label = label[indices]
    print("Data shuffled!")

    split = int(l * 0.8)
    train_input = input[:split]
    train_label = label[:split]
    test_input = input[split:]
    test_label = label[split:]
    print("Data preprocessed!")

    train_input  = train_input
    test_input   = test_input
    train_label = train_label
    test_label  = test_label
    # train_images    = img_prep(data_dict['train_images'])
    # test_images     = img_prep(data_dict['test_images'])
    # word2idx        = data_dict['word2idx']
    # idx2word        = data_dict['idx2word']

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):
        ##############################################################################
        # ## Model Construction
        # decoder_class = {
        #     'rnn'           : RNNDecoder,
        #     'transformer'   : TransformerDecoder
        # }[args.type]

        # decoder = decoder_class(
        #     vocab_size  = len(word2idx), 
        #     hidden_size = args.hidden_size, 
        #     window_size = args.window_size
        # )
        # Vocab size depends on data -- TODO: Save the vocab size when preprocessing, then use it here
        decoder = TransformerDecoder(vocab_size=290, hidden_size=512, window_size=63)
        model = AccompanimentModel(decoder)
        print("Model constructed!")


        # Compile the model
        
        compile_model(model, args)
        print("Model compiled!")
        model_stats = train_model(
            model, train_input, train_label, 0, args, 
            valid = (test_input, test_label)
        )
        with open('src/stats/model_stats.pkl', 'wb') as f:
            pickle.dump(model_stats, f)
        print("Model statistics saved to src/stats/model_stats.pkl")
        print("Model trained!")
        # model.fit(train_input, train_label, batch_size=args.batch_size, epochs=args.epochs)
        if args.chkpt_path: 
            ## Save model to run testing task afterwards
            save_model(model, args)
                
    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        if args.task != 'both': 
            ## Load model for testing. Note that architecture needs to be consistent
            model = load_model(args)
        if not (args.task == 'both' and args.check_valid):
            perp, acc = test_model(model, test_input, test_label, 0, args)
            print(f"Perplexity: {perp}, Accuracy: {acc}")

    ##############################################################################

##############################################################################
## UTILITY METHODS

def save_model(model, args):
    '''Loads model based on arguments'''

    tf.keras.models.save_model(model, args.chkpt_path)
    print(f"Model saved to {args.chkpt_path}")


def load_model(args):
    '''Loads model by reference based on arguments. Also returns said model'''
    model = tf.keras.models.load_model(
        args.chkpt_path,
        custom_objects=dict(
            # AttentionHead           = transformer.AttentionHead,
            # AttentionMatrix         = transformer.AttentionMatrix,
            # MultiHeadedAttention    = transformer.MultiHeadedAttention,
            # TransformerBlock        = transformer.TransformerBlock,
            # PositionalEncoding      = transformer.PositionalEncoding,
            TransformerDecoder      = TransformerDecoder,
            # RNNDecoder              = RNNDecoder,
            AccompanimentModel       = AccompanimentModel
        ),
    )
    
    ## Saving is very nuanced. Might need to set the custom components correctly.
    ## Functools.partial is a function wrapper that auto-fills a selection of arguments. 
    from functools import partial
    model.test    = partial(AccompanimentModel.test,    model)
    model.train   = partial(AccompanimentModel.train,   model)
    model.compile = partial(AccompanimentModel.compile, model)
    compile_model(model, args)
    print(f"Model loaded from '{args.chkpt_path}'")
    return model


def compile_model(model, args):
    '''Compiles model by reference based on arguments'''
    optimizer = tf.keras.optimizers.get(args.optimizer).__class__(learning_rate = args.lr)
    model.compile(
        optimizer   = optimizer,
        loss        = loss_function,
        metrics     = [accuracy_function]
    )


def train_model(model, captions, img_feats, pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    try:
        for epoch in range(args.epochs):
            print("training model!")
            stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
            print("training model done!")
            if args.check_valid:
                print("testing model!")
                stats += [model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)]
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e
        
    return stats


def test_model(model, captions, img_feats, pad_idx, args):
    '''Tests model and returns model statistics'''
    perplexity, accuracy = model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)
    return perplexity, accuracy


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())