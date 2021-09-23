#!/bin/bash

#CUDA_VISIBLE_DEVICES="1" python create_bioasq_run_wSnippets.py warm-donkey-58_val_collection0_doc_r@10
#CUDA_VISIBLE_DEVICES="1" python create_bioasq_run_wSnippets.py zesty-frog-66_val_collection0_doc_map@10

CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_map@10 1
CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_map@10 2
CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_map@10 3
CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_map@10 4

CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_recall@10 1
CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_recall@10 2
CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_recall@10 3
CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_recall@10 4

#CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_map@10
#CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py iconic-wave-1_val_collection0_recall@10

#CUDA_VISIBLE_DEVICES="0" python create_bioasq_run_wSnippets.py earnest-lion-31_val_collection0_map@10
#CUDA_VISIBLE_DEVICES="0" python create_bioasq_run_wSnippets.py earnest-lion-31_val_collection0_recall@10

#CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py dandy-elevator-14_val_collection0_map\@10
#CUDA_VISIBLE_DEVICES="0" python create_bioasq_run.py dandy-elevator-14_val_collection0_recall\@10
