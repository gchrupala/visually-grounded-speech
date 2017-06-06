# Evaluation and analysis

The sections, tables and figures below correspond to the place in the paper
where each result appears and show how to reproduce these results.

## Section 4.3 Image retrieval

### Table 1 and 2

```
python2.7 analyze.py retrieval > retrieval.txt
```

### Figure 2

```
python2.7 analyze.py errors
```
The data will be written to `error-length.txt`.
In order to generate the figure:
```
Rscript error_length.R
```
The plot will be written to `better-length.pdf`.


### Figure 4

```
python2.7 utterance-length.py
```
The plot will be written to `sentlength.pdf`.



### Figure 5

```
python2.7 extract-features.py
python2.7 predict-word-presence.py
```
The plot will be written to `predword.pdf`.


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

### Figure 7

```
python2.7 analyze.py homonyms
```
The data will be written to `ambigu-io.txt` and `ambigu-layerwise.txt`.
In order to generate the figure:

```
Rscript homonyms.R
```
The plot will be written to `ambigu-layerwise.pdf`
