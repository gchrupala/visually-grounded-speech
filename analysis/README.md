# Evaluation and analysis

The sections tables and figures below correspond to the place in the paper
where each result appears and show how to reproduce these results.

## Section 4.3 Image retrieval

### Table 1 and Table 2

```
python2 analyze.py retrieval > retrieval.txt
```

### Figure 2

```
python2 analyze.py errors
```
The data will be written to `error-length.txt`

## Section 4.3 Predicting utterance length

```
python utterance-length.py
```
## Section 4.4 Predicting word presence


```
python predict-word-presence.py
```

## Section 4.5 Sentence similarity

- synthesize-words.py
       generates synthesized version of all words in a dataset.
- synthesize-sentences.py
       generates synthesized version of sentences in a pickled list of sentences as strings

### Figure 6
```
python sentence-similarity.py
Rscript bootstrap-and-plot-correlations.R
```
## Section 4.6 Homonym disambiguation

TBA

### Figure 7
