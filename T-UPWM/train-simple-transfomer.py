import argparse

from mmnrm.utils import set_random_seed, load_neural_model, load_model, load_sentence_generator, flat_list
from nir.embeddings import FastText, Word2Vec

set_random_seed()

import io
from nir.tokenizers import Regex, BioCleanTokenizer, BioCleanTokenizer2, Regex2
import numpy as np
import math
import os 
import json

import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.dataset import TrainCollectionV2, TestCollectionV2, sentence_splitter_builderV2, TrainPairwiseCollection
from mmnrm.modelsv2 import simpletransfomer
from mmnrm.callbacks import TriangularLR, WandBValidationLogger, LearningRateScheduler
from mmnrm.training import PairwiseTraining, pairwise_cross_entropy
from mmnrm.utils import merge_dicts

import nltk

def train_test_generator_for_model(model):
    
    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    max_passages = cfg["max_passages"]
    max_input_size = cfg["max_input_size"]
    tokenizer = model.tokenizer
    
    def maybe_tokenize_pad(query,document):
        if "tokens" not in document:
            input_sentences = []
            sentences =  nltk.sent_tokenize(document["text"])[:max_passages]
            
            for sentence in sentences:
                input_sentences.append([query, sentence])
                
            document["sentences_mask"] = [True]*len(sentences)+[False]*(max_passages-len(sentences))
            
            #pad
            input_sentences.extend([""]*(max_passages-len(sentences)))

            encoded_sentences = tokenizer.batch_encode_plus(
                      input_sentences,
                      max_length=max_input_size,
                      truncation=True,
                      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                      return_token_type_ids=True,
                      padding="max_length",
                      return_attention_mask=True,
                      return_tensors='np',  # Return tf tensors
                )
            document["tokens"] = encoded_sentences
    
    def train_generator(data_generator):
        
        for query, pos_docs, neg_docs in data_generator:
            
            pos_input_ids = []
            pos_input_masks = []
            pos_input_segments = []
            pos_input_mask_sentences = []
                                                                        
            neg_input_ids = []
            neg_input_masks = []
            neg_input_segments = []
            neg_input_mask_sentences = []                                                            
                                                                    
            for i in range(len(query)):
                pos_doc = pos_docs[i]
                neg_doc = neg_docs[i]
                maybe_tokenize_pad(query[i], pos_doc)
                maybe_tokenize_pad(query[i], neg_doc)
                
                pos_input_ids.append(pos_doc["tokens"]["input_ids"])
                pos_input_masks.append(pos_doc["tokens"]["attention_mask"])
                pos_input_segments.append(pos_doc["tokens"]["token_type_ids"])
                pos_input_mask_sentences.append(pos_doc["sentences_mask"]) 
                                                                    
                neg_input_ids.append(neg_doc["tokens"]["input_ids"])
                neg_input_masks.append(neg_doc["tokens"]["attention_mask"])
                neg_input_segments.append(neg_doc["tokens"]["token_type_ids"])
                neg_input_mask_sentences.append(neg_doc["sentences_mask"]) 
                                                                        
            yield  [tf.convert_to_tensor(np.array(pos_input_ids, dtype="int32"), dtype=tf.int32), 
                    tf.convert_to_tensor(np.array(pos_input_masks, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(pos_input_segments, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(pos_input_mask_sentences, dtype="bool"), dtype=tf.bool)],\
                   [tf.convert_to_tensor(np.array(neg_input_ids, dtype="int32"), dtype=tf.int32), 
                    tf.convert_to_tensor(np.array(neg_input_masks, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(neg_input_segments, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(neg_input_mask_sentences, dtype="bool"), dtype=tf.bool)]
    
    def test_generator(data_generator):
        
        for ids, queries, docs in data_generator:
        
            input_query_ids = []

            input_ids = []
            input_masks = []
            input_segments = []

            input_mask_sentences = []
            docs_ids = []

            for i in range(len(ids)):
                for doc in docs[i]:
                    maybe_tokenize_pad(queries[i], doc)
                    input_mask_sentences.append(doc["sentences_mask"])
                    input_ids.append(doc["tokens"]["input_ids"])
                    input_masks.append(doc["tokens"]["attention_mask"])
                    input_segments.append(doc["tokens"]["token_type_ids"])
                    docs_ids.append(doc["id"])
                    input_query_ids.append(ids[i])

            yield input_query_ids, [tf.convert_to_tensor(np.array(input_ids, dtype="int32"), dtype=tf.int32), 
                                    tf.convert_to_tensor(np.array(input_masks, dtype="int32"), dtype=tf.int32),
                                    tf.convert_to_tensor(np.array(input_segments, dtype="int32"), dtype=tf.int32),
                                    tf.convert_to_tensor(np.array(input_mask_sentences, dtype="bool"), dtype=tf.bool)], docs_ids, None
    
    return train_generator, test_generator



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This is program will perform pairwise training')
    parser.add_argument('base_lr', type=float, help="base learning rate")
    parser.add_argument('-use_tlr', action='store_true')
    parser.add_argument('-use_stepdecay', action='store_true')
    parser.add_argument('-max_passages', default=20)
    parser.add_argument('-max_input_size', default=128)
    
    args = parser.parse_args()
    
    use_triangularLR = args.use_tlr
    use_step_decay = args.use_stepdecay
    
    LR = args.base_lr
    base_lr = 0.001 if args.base_lr>0.001 else args.base_lr
    max_lr = 0.01
    epoch=20
    
    train_batch_size=32
    
    cfg = {
        "checkpoint_name":'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        "model":{
            "max_passages" : int(args.max_passages),
            "max_input_size" : int(args.max_input_size),
        }
    }
    
    ranking_model = simpletransfomer(**cfg)
    
    train_input_generator, test_input_generator = train_test_generator_for_model(ranking_model)
    
    training_data_used = "training_batch_03_0.4_0.14_250"
    train_collection = TrainCollectionV2\
                                .load(training_data_used)\
                                .batch_size(train_batch_size)\
                                .set_transform_inputs_fn(train_input_generator)

    
    validation_collection = TestCollectionV2.load("validation_batch_03_0.5_0.79_100")\
                                    .batch_size(32)\
                                    .set_transform_inputs_fn(test_input_generator)\
                                    .set_name("Validation TOP 100 k1 0.5 0.79")
    
    validation_collection_2 = TestCollectionV2.load("validation_batch_03_0.9_0.09_100")\
                                    .batch_size(32)\
                                    .set_transform_inputs_fn(test_input_generator)\
                                    .set_name("Validation TOP 100 k1 0.9 0.09")
    
    if use_triangularLR:
        _lr = "tlr_"+str(base_lr)+"_"+str(max_lr)
    elif use_step_decay:
        _lr = "step_decay_"+str(LR)
    else:
        _lr = LR
        
    wandb_config = {"optimizer": "adam",
                     "lr":_lr,
                     "loss":"pairwise_cross_entropy",
                     "train_batch_size":train_batch_size,
                     "epoch":epoch,
                     "name": "sibm transformer",
                     "training_dataset": training_data_used,
                     "checkpoint_name": cfg["checkpoint_name"]
                     }
    
    wandb_config = merge_dicts(wandb_config, cfg["model"])
    
    project_name = "bioasq-9b-rnd3"
    
    wandb_args = {"project": project_name, "config": wandb_config}
    
    tlr = TriangularLR(base_lr=base_lr, max_lr=max_lr,)
    
    wandb_val_logger = WandBValidationLogger(wandb_args=wandb_args,
                                             steps_per_epoch=train_collection.get_steps(),
                                             validation_collection=[validation_collection, validation_collection_2],
                                             test_collection=None,
                                             path_store = "trained_models",
                                             output_metrics=["map@10",
                                                             "recall@10",
                                                             ])
    
    step_decay_lr = LearningRateScheduler(initial_learning_rate=LR)

    train_callbacks = [wandb_val_logger]
    
    if use_triangularLR:
        train_callbacks.append(tlr)

    if use_step_decay:
        train_callbacks.append(step_decay_lr)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
    
    @tf.function
    def clip_grads(grads):
        gradients, _ = tf.clip_by_global_norm(grads, 5.0)
        return gradients
    
    ranking_model.summary()
    
    train = PairwiseTraining(model=ranking_model,
                             train_collection=train_collection,
                             loss=pairwise_cross_entropy,
                             grads_callback=clip_grads,
                             optimizer=optimizer,
                             callbacks=train_callbacks)

                              
    train.train(epoch, draw_graph=False)
    
    