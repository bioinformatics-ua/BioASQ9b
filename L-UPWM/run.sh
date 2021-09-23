py-bioasq/bin/python train.py 0.005 -tokenizer 1
py-bioasq/bin/python train.py 0.01
py-bioasq/bin/python train.py 0.001 -use_tlr
py-bioasq/bin/python train.py 0.001 -max_q_terms 25 -max_p_terms 40
py-bioasq/bin/python train.py 0.001 -use_query_sw -use_docs_sw