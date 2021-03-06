# visually-grounded-speech

This repository contains code to reproduce the results from: 
- Chrupała, G., Gelderloos, L., & Alishahi, A. (2017). Representations of language in a model of visually grounded speech signal. ACL. arXiv preprint: https://arxiv.org/abs/1702.01991

## Installation

First, download and install funktional version 0.6: https://github.com/gchrupala/funktional/releases/tag/0.6

Second, install the code in the current repo:

    python setup.py develop

You also need to download and unpack the files `data.tgz` and `models.tgz` from http://doi.org/10.5281/zenodo.804392.
The files in `models.tgz` contain the pre-trained models used for the analyses in the paper.

After unpacking these files you should have the directories `data` and `models`.


For the scripts in [analysis](analysis), you also need to install:

- sklearn
- pandas
- matplotlib
- keras


## Usage

### Training models

In order to re-train one of the models, change to the corresponding directory in [experiments](experiments), and execute:

```
python2.7 run.py > log.txt
```
### Analysis

See [analysis/README.md](analysis/README.md)

