# Evaluation and analysis

The sections, tables and figures below correspond to the place in the paper
where each result appears and show how to reproduce these results.

## Section 4.2 Image retrieval

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


## Section 4.3 Predicting utterance length

### Figure 4

In order to generate the figure:
```
python2.7 utterance-length.py
```
The plot will be written to `sentlength.pdf`.


## Section 4.4 Predicting word presence

### Figure 5

In order to generate the figure:
```
python2.7 predict-word-presence.py
```
The plot will be written to `predword.pdf`.

Pre-extracted feature files for this experiment are included in `data.tgz`. If you need to re-generate them, run:

```
python2.7 extract-features.py
```
The features will be written to:
- `../data/coco/words-coco.npy`
- `../data/coco/mfcc-coco.npy`
- `../data/flickr8k/words-flickr8k.npy`
- `../data/flickr8k/mfcc-flickr8k.npy`

## Section 4.5 Sentence similarity

### Figure 6

In order to create the figure, run:

```
python2.7 sentence-similarity.py
Rscript bootstrap-and-plot-correlations.R
```
Figure will be save in `xxx.pdf`

Pre-extracted feature files for this experiment are included in `data.tgz`. If you need to re-generate them, run:

```
python2.7 zzz.py
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
