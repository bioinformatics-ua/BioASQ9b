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

from mmnrm.dataset import TrainCollectionV2, TestCollectionV2, sentence_splitter_builderV2, TrainPairwiseCollection, TrainSnippetsCollectionV2
from mmnrm.modelsv2 import sibmtransformer_wSnippets
from mmnrm.callbacks import TriangularLR, WandBValidationLogger, LearningRateScheduler
from mmnrm.training import PairwiseTraining, pairwise_cross_entropy
from mmnrm.utils import merge_dicts
from nltk.tokenize.punkt import PunktSentenceTokenizer

import nltk

def test_generator_for_model(model):
    
    punkt_sent_tokenizer = PunktSentenceTokenizer().span_tokenize
    
    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    max_passages = cfg["max_passages"]
    max_input_size = cfg["max_input_size"]
    tokenizer = model.tokenizer
    
    def maybe_tokenize_pad(query, document):
        if "tokens" not in document:
            input_sentences = []
            
            sentences = []
            spans = []
            for _itter, position in enumerate(list(punkt_sent_tokenizer(document["text"]))[:max_passages]):
                start, end = position
                _text = document["text"][start:end]

                is_title = True
                if _itter>0: # fix the start and end position for the abstract

                    if _itter == 1: # auxiliar correction to set the abstract at 0 index
                        diff = (len(document["title"])-1)+(start-(len(document["title"])-1))

                    start = start-diff
                    end = end-diff
                    is_title = False

                sentences.append(_text)
                spans.append({"start":start,
                              "end":end-1,
                              "text":_text,
                              "is_title":is_title,
                              "snippet_id":document["id"]+"_"+str(_itter),
                              "doc_id":document["id"]})
            document["spans"] = spans
            
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
    
    def test_generator(data_generator):
        
        for ids, queries, docs in data_generator:
        
            input_query_ids = []

            input_ids = []
            input_masks = []
            input_segments = []

            input_mask_sentences = []
            docs_ids = []
            docs_spans = []

            for i in range(len(ids)):
                for doc in docs[i]:
                    maybe_tokenize_pad(queries[i], doc)
                    input_mask_sentences.append(doc["sentences_mask"])
                    input_ids.append(doc["tokens"]["input_ids"])
                    input_masks.append(doc["tokens"]["attention_mask"])
                    input_segments.append(doc["tokens"]["token_type_ids"])
                    docs_ids.append(doc["id"])
                    docs_spans.append(doc["spans"])
                    input_query_ids.append(ids[i])

            yield input_query_ids, [tf.convert_to_tensor(np.array(input_ids, dtype="int32"), dtype=tf.int32), 
                                    tf.convert_to_tensor(np.array(input_masks, dtype="int32"), dtype=tf.int32),
                                    tf.convert_to_tensor(np.array(input_segments, dtype="int32"), dtype=tf.int32),
                                    tf.convert_to_tensor(np.array(input_mask_sentences, dtype="bool"), dtype=tf.bool)], docs_ids, docs_spans
    
    return test_generator


def train_test_generator_for_model(model):
    
    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    max_passages = cfg["max_passages"]
    max_input_size = cfg["max_input_size"]
    tokenizer = model.tokenizer
    
    punkt_sent_tokenizer = PunktSentenceTokenizer().span_tokenize
    
    pad_labels = lambda x, max_lim, dtype='int32': x[:max_lim] + [0]*(max_lim-len(x))
    
    def sent_tokenize(document):
        return [ document[start:end] for start,end in punkt_sent_tokenizer(document) ]
    
    def maybe_tokenize_pad(query, document, train=False, labels=None):
        
        if "tokens" not in document:
            input_sentences = []
            if not train:
                sentences = sent_tokenize(document["text"])[:max_passages]
            else:
                sentences = list(map(lambda x: x["text"],document["snippets"]))[:max_passages]
                
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
            
            if labels is not None:
                document["sentences_labels"] = pad_labels(labels, cfg["max_passages"])
    
    def train_generator(data_generator):
        
        for query, pos_docs, pos_label, neg_docs in data_generator:
            
            pos_input_ids = []
            pos_input_masks = []
            pos_input_segments = []
            pos_snippets_labels = []
            pos_input_mask_sentences = []
                                                                        
            neg_input_ids = []
            neg_input_masks = []
            neg_input_segments = []
            neg_input_mask_sentences = []                                                            
                                                                    
            for i in range(len(query)):
                pos_doc = pos_docs[i]
                neg_doc = neg_docs[i]
                maybe_tokenize_pad(query[i], pos_doc, True, pos_label[i])
                maybe_tokenize_pad(query[i], neg_doc, True)
                
                pos_input_ids.append(pos_doc["tokens"]["input_ids"])
                pos_input_masks.append(pos_doc["tokens"]["attention_mask"])
                pos_input_segments.append(pos_doc["tokens"]["token_type_ids"])
                pos_snippets_labels.append(pos_doc["sentences_labels"])
                pos_input_mask_sentences.append(pos_doc["sentences_mask"]) 
                                                                    
                neg_input_ids.append(neg_doc["tokens"]["input_ids"])
                neg_input_masks.append(neg_doc["tokens"]["attention_mask"])
                neg_input_segments.append(neg_doc["tokens"]["token_type_ids"])
                neg_input_mask_sentences.append(neg_doc["sentences_mask"]) 
                                                                        
            yield  [tf.convert_to_tensor(np.array(pos_input_ids, dtype="int32"), dtype=tf.int32), 
                    tf.convert_to_tensor(np.array(pos_input_masks, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(pos_input_segments, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(pos_input_mask_sentences, dtype="bool"), dtype=tf.bool),
                    tf.convert_to_tensor(np.array(pos_snippets_labels, dtype="float32"), dtype=tf.float32)],\
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
    parser.add_argument('-use_sentence_scores', action='store_true')
    parser.add_argument('-use_sentence_unbound', action='store_true')
    parser.add_argument('-max_passages', default=20)
    parser.add_argument('-max_input_size', default=128)
    parser.add_argument('-top_k_list', nargs="*", default=[3, 5, 10, 15])
    parser.add_argument('-match_threshold', default=0.9)
    parser.add_argument('-gamma', default=0.5)
    
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
            "match_threshold" : float(args.match_threshold),
            "apriori_exact_match" : True,
            "use_sentence_unbound": args.use_sentence_unbound,
            "use_transformer_sentence_scores" : args.use_sentence_scores,
            "sentence_hidden_size": None,
        }
    }
    
    ranking_model = sibmtransformer_wSnippets(**cfg)
    
    train_input_generator, test_input_generator = train_test_generator_for_model(ranking_model)
    test_input_wSnippets = test_generator_for_model(ranking_model)
    
    training_data_used = "joint_training_hardlabel_batch_05_0.6_0.51_250"
    train_collection = TrainSnippetsCollectionV2\
                                .load(training_data_used)\
                                .batch_size(train_batch_size)\
                                .set_transform_inputs_fn(train_input_generator)

    
    validation_collection = TestCollectionV2.load("validation_wSnippets_batch_05_0.4_0.44_100")\
                                    .batch_size(100)\
                                    .set_transform_inputs_fn(test_input_wSnippets)\
                                    .set_name("Validation TOP (recall) 100")

    validation_collection_2 = TestCollectionV2.load("validation_batch_05_0.5_0.15_100")\
                                    .batch_size(32)\
                                    .set_transform_inputs_fn(test_input_generator)\
                                    .set_name("Validation TOP (map) 100")
    
    if use_triangularLR:
        _lr = "tlr_"+str(base_lr)+"_"+str(max_lr)
    elif use_step_decay:
        _lr = "step_decay_"+str(LR)
    else:
        _lr = LR
    
    gamma = float(args.gamma)
    
    wandb_config = {"optimizer": "adam",
                     "lr":_lr,
                     "gamma": gamma,
                     "loss":"pairwise_cross_entropy",
                     "train_batch_size":train_batch_size,
                     "epoch":epoch,
                     "name": "sibm transformer",
                     "training_dataset": training_data_used,
                     "checkpoint_name": cfg["checkpoint_name"]
                     }
    
    wandb_config = merge_dicts(wandb_config, cfg["model"])
    
    project_name = "bioasq-9b-rnd5"
    
    wandb_args = {"project": project_name, "config": wandb_config}
    
    tlr = TriangularLR(base_lr=base_lr, max_lr=max_lr,)
    
    def snippetRank_byThreshold(threshold):
    
        def snippetRank(results):

            snippets_results = {}
            # this will follow the document order first
            for q in results.keys():
                snippets_results[q] = [y for y in flat_list([x["snippets"] for x in results[q]]) if y["score"] >= threshold]

            return snippets_results

        return snippetRank
    
    snippet_rank_f = snippetRank_byThreshold(0.08) 
    
    wandb_val_logger = WandBValidationLogger(wandb_args=wandb_args,
                                             steps_per_epoch=train_collection.get_steps(),
                                             validation_collection=[validation_collection, validation_collection_2],
                                             test_collection=None,
                                             snippet_rank_f=snippet_rank_f,
                                             path_store = "trained_models",
                                             output_metrics=["map@10",
                                                             "recall@10",
                                                             "doc_r@10", 
                                                             "doc_map@10",
                                                             "snippet_f1@10",
                                                             "snippet_map@10"
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
    
    def transform_inputs(pos_in, neg_in):
        return pos_in[:4], neg_in, pos_in[4], None
    
    
    def transformations_for_pairwise(vector):
        
        vector_t = tf.transpose(vector, perm=[0,2,1])
        vector_repeat = tf.repeat(vector, cfg["model"]["max_passages"], axis=-1)
        vector_t_repeat = tf.repeat(vector_t, cfg["model"]["max_passages"], axis=-2)
        
        vector_xor = tf.cast(tf.math.logical_xor(tf.cast(vector_repeat, tf.bool), 
                                                              tf.cast(vector_t_repeat, tf.bool)), 
                                          tf.float32)
        
        vector_repeat = tf.cast(vector_repeat, tf.float32)
        vector_t_repeat = tf.cast(vector_t_repeat, tf.float32)
        
        return vector_repeat, vector_t_repeat, vector_xor
        
    @tf.function
    def joint_loss_pairwise(pos, neg, pos_label, neg_label):
        
        pos_score = pos[0]
        neg_score = neg[0]

        p_wise_loss = pairwise_cross_entropy(pos_score, neg_score)
        print(p_wise_loss)
        ## PAIRWISE SNIPPET SCORE LOSS
        
        pos_sentence_labels = tf.cast(tf.reshape(pos_label, (-1, cfg["model"]["max_passages"], 1)), tf.int32)
        pos_sentence_scores = tf.cast(tf.math.exp(tf.reshape(pos[1], (-1, cfg["model"]["max_passages"], 1))), tf.float32)
        
        # labels
        pos_sentence_labels_repeat, pos_sentence_labels_repeat_transpose, pos_sentence_lables_xor =  transformations_for_pairwise(pos_sentence_labels)

        # scores
        pos_sentence_scores_repeat, pos_sentence_scores_repeat_transpose, pos_sentence_scores_xor =  transformations_for_pairwise(pos_sentence_scores)
        
        snippet_loss_numerator = pos_sentence_lables_xor*(pos_sentence_labels_repeat*pos_sentence_scores_repeat + pos_sentence_labels_repeat_transpose*pos_sentence_scores_repeat_transpose)

        snippet_loss_denominator = pos_sentence_scores_repeat*pos_sentence_lables_xor + pos_sentence_lables_xor*pos_sentence_scores_repeat_transpose + 0.0000001
        """
        snippet_loss = tf.math.log(snippet_loss_numerator/snippet_loss_denominator)
        
        snippet_loss = tf.reshape(snippet_loss, (-1,model_cfg["max_passages"]*model_cfg["max_passages"]))
        
        snippet_loss_mask = tf.cast(snippet_loss>-5, tf.float32)
        
        snippet_loss = -snippet_loss
        
        snippet_loss_sum = tf.math.reduce_sum(snippet_loss*snippet_loss_mask, axis=-1)
        snippet_loss_num = tf.math.reduce_sum(snippet_loss_mask,  axis=-1)
        

        snippet_loss = snippet_loss_sum/(snippet_loss_num+ 0.0000001)

        snippet_loss = tf.math.reduce_mean(snippet_loss)
        """
        
        snippet_loss = snippet_loss_numerator/snippet_loss_denominator
        snippet_loss = tf.reshape(snippet_loss, (-1,))
        
        snippet_loss_mask = snippet_loss>0.01
        snippet_loss_indices = tf.cast(tf.where(snippet_loss_mask), tf.int32)
        
        snippet_loss = tf.gather_nd(snippet_loss, snippet_loss_indices)
        
        snippet_loss = -tf.math.log(snippet_loss)
        
        snippet_loss = tf.math.reduce_mean(snippet_loss)
        
        return (gamma * p_wise_loss) + ((1-gamma) * snippet_loss)
    
    @tf.function
    def joint_loss(pos, neg, pos_label, neg_label):

        pos_score = pos[0]
        neg_score = neg[0]

        p_wise_loss = pairwise_cross_entropy(pos_score, neg_score) 
        
        pos_sentence_scores = tf.reshape(pos[1], (-1, cfg["model"]["max_passages"]))
        pos_sentence_labels = tf.reshape(pos_label, (-1, cfg["model"]["max_passages"]))
        
        pos_xentropy = tf.keras.losses.binary_crossentropy(pos_sentence_labels, pos_sentence_scores) 
        
        snippet_loss = K.mean(pos_xentropy, axis=-1)
        
        return (gamma * p_wise_loss) + ((1-gamma) * snippet_loss)# + neg_snippet_loss
    
    ranking_model.summary()
    
    train = PairwiseTraining(model=ranking_model,
                             train_collection=train_collection,
                             transform_model_inputs_callback=transform_inputs,
                             loss=joint_loss_pairwise,
                             grads_callback=clip_grads,
                             optimizer=optimizer,
                             callbacks=train_callbacks)

                              
    train.train(epoch, draw_graph=False)
    
    