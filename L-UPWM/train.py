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
from mmnrm.modelsv2 import sibm
from mmnrm.callbacks import TriangularLR, WandBValidationLogger, LearningRateScheduler
from mmnrm.training import PairwiseTraining, pairwise_cross_entropy
from mmnrm.utils import merge_dicts


import nltk

def build_data_generators(tokenizer, queries_sw=None, docs_sw=None):
    
    def maybe_tokenize(documents):
        if "tokens" not in documents:
            split = nltk.sent_tokenize(documents["text"])
            documents["tokens"] = tokenizer.texts_to_sequences(split)
            if docs_sw is not None:
                for tokenized_sentence in documents["tokens"]:
                    tokenized_sentence = [token for token in tokenized_sentence if token not in docs_sw]
    
    def train_generator(data_generator):
        while True:

            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)

            # tokenization, this can be cached for efficientcy porpuses NOTE!!
            tokenized_query = tokenizer.texts_to_sequences(query)

            if queries_sw is not None:
                for tokens in tokenized_query:
                    tokenized_query = [token for token in tokens if token not in queries_sw] 
            
            saveReturn = True
            
            for batch_index in range(len(pos_docs)):
                
                # tokenizer with cache in [batch_index][tokens]
                maybe_tokenize(pos_docs[batch_index])
                
                # assertion
                if all([ len(sentence)==0  for sentence in pos_docs[batch_index]["tokens"]]):
                    saveReturn = False
                    break # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                          # for obvious reasons
                
                maybe_tokenize(neg_docs[batch_index])
                
            if saveReturn: # this is not true, if the batch is rejected
                yield tokenized_query, pos_docs, neg_docs

    def test_generator(data_generator):
        for _id, query, docs in data_generator:
            tokenized_queries = []
            for i in range(len(_id)):
                # tokenization
                tokenized_query = tokenizer.texts_to_sequences([query[i]])[0]

                if queries_sw is not None:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
                
                tokenized_queries.append(tokenized_query)
                    
        
                for doc in docs[i]:
                    maybe_tokenize(doc)
                                                 
            yield _id, tokenized_queries, docs
            
    return train_generator, test_generator

def model_train_generator_for_model(model):

    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    train_gen, test_gen = build_data_generators(model.tokenizer)
    
    pad_tokens = lambda x, max_len, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_len,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_sentences = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))
    
    def maybe_padding(document):
        if isinstance(document["tokens"], list):
            document["tokens"] = pad_tokens(pad_sentences(document["tokens"], cfg["max_passages"]), cfg["max_p_terms"])
            
    def train_generator(data_generator):
 
        for query, pos_docs, neg_docs in train_gen(data_generator):
            
            query = pad_tokens(query, cfg["max_q_terms"])
            
            pos_docs_array = []
            neg_docs_array = []
            
            # pad docs, use cache here
            for batch_index in range(len(pos_docs)):
                maybe_padding(pos_docs[batch_index])
                pos_docs_array.append(pos_docs[batch_index]["tokens"])
                maybe_padding(neg_docs[batch_index])
                neg_docs_array.append(neg_docs[batch_index]["tokens"])
            
            yield [query, np.array(pos_docs_array)], [query, np.array(neg_docs_array)]
            
    def test_generator(data_generator):
        
        for ids, query, docs in test_gen(data_generator):
            
            docs_ids = []
            docs_array = []
            query_array = []
            query_ids = []
            
            for i in range(len(ids)):
                
                for doc in docs[i]:
                    # pad docs, use cache here
                    maybe_padding(doc)
                    docs_array.append(doc["tokens"])
                    docs_ids.append(doc["id"])
                
                query_tokens = pad_tokens([query[i]], cfg["max_q_terms"])[0]
                query_tokens = [query_tokens] * len(docs[i])
                query_array.append(query_tokens)
                    
                query_ids.append([ids[i]]*len(docs[i]))
            
            yield flat_list(query_ids), [np.array(flat_list(query_array)), np.array(docs_array)], docs_ids, None
            
    return train_generator, test_generator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This is program will perform pairwise training')
    parser.add_argument('base_lr', type=float, help="base learning rate")
    parser.add_argument('-use_query_sw', action='store_true')
    parser.add_argument('-use_docs_sw', action='store_true')
    parser.add_argument('-use_tlr', action='store_true')
    parser.add_argument('-use_stepdecay', action='store_true')
    parser.add_argument('-max_passages', default=20)
    parser.add_argument('-max_q_terms', default=30)
    parser.add_argument('-max_p_terms', default=30)
    parser.add_argument('-filters', default=16)
    parser.add_argument('-top_k_list', nargs="*", default=[3, 5, 10, 15])
    parser.add_argument('-match_threshold', default=0.99)
    parser.add_argument('-tokenizer', default=0, type=int)

    
    args = parser.parse_args()
    
    min_freq = 0
    mun_itter = 15
    emb_size = 200
    
    use_triangularLR = args.use_tlr
    use_step_decay = args.use_stepdecay

    LR = args.base_lr
    base_lr = 0.001 if args.base_lr>0.001 else args.base_lr
    max_lr = 0.01
    epoch=32

    train_batch_size=32
    type_split_mode=4
    use_query_sw = args.use_query_sw
    use_docs_sw = args.use_docs_sw

    cache_folder = "/backup/BioASQ-9b"
    index_name = "bioasq_9b"
    print("build new model")

    # build config
    if args.tokenizer==0:
        tokenizer_class = Regex
    elif args.tokenizer==1:
        tokenizer_class = Regex2
    else:
        raise ValueError(f"Tokenizer {tokenizer_class} unvailable")
        
    tokenizer_cfg = {"class":tokenizer_class,
                    "attr":{
                        "cache_folder": os.path.join(cache_folder, "tokenizers"),
                        "prefix_name": index_name
                    },
                    "min_freq":min_freq}

    embeddind_class = Word2Vec
    embedding_cfg = {
        "class":embeddind_class,
        "attr":{
            "cache_folder": os.path.join(cache_folder, "embeddings"),
            "prefix_name":index_name,
            "path":"/backup/pre-trained_embeddings/word2vec/"+index_name+"_gensim_iter_"+str(mun_itter)+"_freq"+str(min_freq)+"_"+str(emb_size)+"_"+tokenizer_class.__name__+"_word2vec.bin",
        }
    }

    model_cfg = {
        "max_q_terms": int(args.max_q_terms),
        "max_passages": int(args.max_passages),
        "max_p_terms": int(args.max_p_terms),
        "filters": int(args.filters),
        "match_threshold": float(args.match_threshold),
        "activation": "mish",
        "top_k_list": list(args.top_k_list),
        "use_avg_pool":True,
        "use_kmax_avg_pool":True,
        
        "semantic_normalized_query_match" : False,
        "return_snippets_score" : False
    }
    
    print(list(args.top_k_list))
    
    cfg = {"model":model_cfg, "tokenizer": tokenizer_cfg, "embedding": embedding_cfg}


    K.clear_session()

    ranking_model = sibm(**cfg)

    load_pretrained_model = "new model"

    train_input_generator, test_input_generator = model_train_generator_for_model(ranking_model)



            
    #Training and validation data 
            
    
    training_data_used = "training_batch_01_250"
    train_collection = TrainCollectionV2\
                                .load(training_data_used)\
                                .batch_size(train_batch_size)\
                                .set_transform_inputs_fn(train_input_generator)

    
    validation_collection = TestCollectionV2.load("validation_data_batch_01_250")\
                                    .batch_size(250)\
                                    .set_transform_inputs_fn(test_input_generator)\
                                    .set_name("Validation TOP 250")

    
    notes = ""

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
                     "name": "sibm model",
                     "query_sw":use_query_sw,
                     "docs_sw":use_docs_sw,
                     "load_pretrained_model":load_pretrained_model,
                     "training_dataset": training_data_used,
                     "tokenizer_class": str(tokenizer_class),
                     "notes": notes
                     }

    wandb_config = merge_dicts(wandb_config, model_cfg)


    project_name = "bioasq-9b-rnd1"

    ## config wandb
    wandb_args = {"project": project_name, "config": wandb_config}

    
    tlr = TriangularLR(base_lr=base_lr, max_lr=max_lr,)

    wandb_val_logger = WandBValidationLogger(wandb_args=wandb_args,
                                             steps_per_epoch=train_collection.get_steps(),
                                             validation_collection=[validation_collection],
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
    
    
    train = PairwiseTraining(model=ranking_model,
                             train_collection=train_collection,
                             loss=pairwise_cross_entropy,
                             grads_callback=clip_grads,
                             optimizer=optimizer,
                             callbacks=train_callbacks)

                              
    train.train(epoch, draw_graph=False)  