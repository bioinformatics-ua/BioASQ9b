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
from mmnrm.modelsv2 import sibmtransfomer
from mmnrm.callbacks import TriangularLR, WandBValidationLogger, LearningRateScheduler
from mmnrm.training import PairwiseTraining, pairwise_cross_entropy
from mmnrm.utils import merge_dicts, load_model

from timeit import default_timer as timer

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
                                                                        
            yield  [np.array(pos_input_ids, dtype="int32"), 
                    np.array(pos_input_masks, dtype="int32"),
                    np.array(pos_input_segments, dtype="int32"),
                    np.array(pos_input_mask_sentences, dtype="bool")],\
                   [np.array(neg_input_ids, dtype="int32"), 
                    np.array(neg_input_masks, dtype="int32"),
                    np.array(neg_input_segments, dtype="int32"),
                    np.array(neg_input_mask_sentences, dtype="bool")]
    
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

            yield input_query_ids, [np.array(input_ids, dtype="int32"), 
                                    np.array(input_masks, dtype="int32"),
                                    np.array(input_segments, dtype="int32"),
                                    np.array(input_mask_sentences, dtype="bool")], docs_ids, None
    
    return train_generator, test_generator


if __name__ == "__main__":
    
    
    
    rank_model = load_model("trained_models/splendid-brook-38_val_collection0_map@10")

    rank_model.summary()
    
    _, test_input_generator = train_test_generator_for_model(rank_model)
    
    def convert_to_tensor(data_generator):
        for query_id, Y, docs_ids, offset in test_input_generator(data_generator):
            yield query_id, [tf.convert_to_tensor(Y[0], dtype=tf.int32), tf.convert_to_tensor(Y[1], dtype=tf.int32), tf.convert_to_tensor(Y[2], dtype=tf.int32), tf.convert_to_tensor(Y[3], dtype=tf.bool)], docs_ids, offset
    
    data_generator = TestCollectionV2.load("validation_data_batch_01_25")\
                                    .batch_size(32)\
                                    .set_transform_inputs_fn(convert_to_tensor)\
                                    .set_name("Validation TOP 25")
    
    
    from collections import defaultdict
    import time 

    generator_Y = data_generator.generator()
    q_scores = defaultdict(list)

    @tf.function
    def run(x):
        return rank_model(x)

    for i, _out in enumerate(generator_Y):
        query_id, Y, docs_ids, _ = _out
        s_time = time.time()
        scores = rank_model(Y)[:,0]
        if not i%50:
            print("\rEvaluation {} | avg-time {}".format(i, time.time()-s_time), end="\r")

        for i in range(len(scores)):

            #q_scores[query_id].extend(list(zip(docs_ids,scores)))
            #q_scores[query_id[i]].append({"id":docs_info[i],
            #                              "score":scores[i]})
            q_scores[query_id[i]].append((docs_ids[i],scores[i]))

    # sort the rankings
    for query_id in q_scores.keys():
        q_scores[query_id].sort(key=lambda x:-x[1])
    
    #print(q_scores)
    
    print(data_generator.evaluate(q_scores))