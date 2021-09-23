<h1 align="center">
  BioASQ 9b - phase A(2021)
</h1>

This repo is divided into 3 main folders, the Baseline, L-UPWM and T-UPWM.

The Baseline contains the code associated with the BM25 runs, which corresponds to the first stage of retrieval. Both L-UPWM and T-UPWM use the baseline runs from here.
The L-UPWM and T-UPWM both correspond to our reranking solutions.

This code is mainly offered as a way to visualize the step that we performed, however, anyone can run it with the additional care to change all of the hardcoded paths on the notebooks and scripts. However, it is advised to first run the setup.sh script that is inside of each folder to correctly install all the dependencies in a virtual environment 

### UPWM

Lastly, the UPWM in here is sometimes called of SIBM, which is an old name. We are in process to extract the UPWM to a isolated repo, where anyone can used by just decorating any Keras model!.... stay tuned

### Team
  * Tiago Almeida<sup id="a1">[1](#f1)</sup>
  * Sérgio Matos<sup id="a1">[1](#f1)</sup>

1. <small id="f1"> University of Aveiro, Dept. Electronics, Telecommunications and Informatics (DETI / IEETA), Aveiro, Portugal </small> [↩](#a1)

### Cite

```bib
@inproceedings{DBLP:conf/clef/AlmeidaM21a,
  author    = {Tiago Almeida and
               S{\'{e}}rgio Matos},
  editor    = {Guglielmo Faggioli and
               Nicola Ferro and
               Alexis Joly and
               Maria Maistro and
               Florina Piroi},
  title     = {Universal Passage Weighting Mecanism {(UPWM)} in BioASQ 9b},
  booktitle = {Proceedings of the Working Notes of {CLEF} 2021 - Conference and Labs
               of the Evaluation Forum, Bucharest, Romania, September 21st - to -
               24th, 2021},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2936},
  pages     = {196--212},
  publisher = {CEUR-WS.org},
  year      = {2021},
  url       = {http://ceur-ws.org/Vol-2936/paper-13.pdf},
  timestamp = {Tue, 31 Aug 2021 14:51:15 +0200},
  biburl    = {https://dblp.org/rec/conf/clef/AlmeidaM21a.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
