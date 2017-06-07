import cPickle as pickle
import csv
import numpy
import string
import sys

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr, zscore

import imaginet.task as task
import imaginet.tts as tts
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.defn.visual2_rhn as vis2


def tokenize(sent):
    s = sent.translate(string.maketrans("",""), string.punctuation)
    s = s.lower()
    words = s.split()
    return words

def cosine_similarity(a, b):
    # returns cosine smilarity between a and b
    return 1.0-cosine(a, b)

def z_score(a, b):
    """
    uses scipy.stats.zscore to transform to z-scores
    expects two arrays corresponding to sentences a and sentences b
    of sentence pairs, and returns two arrays which correspond to
    a and b undergone the same scaling by the mean & SD of their
    concatenation
    """
    if len(a) != len(b):
        print "warning: uneven nr of sentences in a and b"
    all_sents = numpy.concatenate((a, b))
    scaled = zscore(all_sents)
    return scaled[:len(a),:], scaled[len(a):,:]

def cosine_similarities(a, b, transform):
    """
    returns list of cosine similarities between lists of vectors
    a and b. The z_score transformation is applied if transform == True
    """
    a = numpy.stack(a)
    b = numpy.stack(b)
    #transform if requested
    if transform:
        print "transforming"
        # z_score is written to apply same scale to a and b
        a, b = z_score(a, b)
    print "calculating cosine dists"
    cos = [cosine_similarity(a[i], b[i]) for i in range(len(a))]
    return cos

def get_batch_mfccs(batch):
    with open("speech/sentences_a_{}.pkl".format(batch), "rb") as f:
        audio_a = pickle.load(f)
    mfcc_a = [ tts.extract_mfcc(audio) for audio in audio_a ]
    with open("speech/sentences_b_{}.pkl".format(batch), "rb") as f:
        audio_b = pickle.load(f)
    mfcc_b = [ tts.extract_mfcc(audio) for audio in audio_b ]
    return mfcc_a, mfcc_b

def load_batch_mfccs(batch):
    mfcc_a = numpy.load("../data/coco/sick/mfccs/a_{}.npy".format(batch))
    mfcc_b = numpy.load("../data/coco/sick/mfccs/b_{}.npy".format(batch))
    return mfcc_a, mfcc_b

# copied implementation from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)
    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)
    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = numpy.array(tuple(source))
    target = numpy.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = numpy.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1
        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = numpy.minimum(current_row[1:], numpy.add(previous_row[:-1], target != s))
        # Deletion (target grows shorter than source):
        current_row[1:] = numpy.minimum(current_row[1:], current_row[0:-1] + 1)
        previous_row = current_row
    return previous_row[-1]

# returns levenshtein distance divided by length of longest item
def norm_levenshtein(item1, item2):
    lev = float(levenshtein(item1, item2))
    maxlev = float(max(len(item1), len(item2)))
    return lev/maxlev

############################

# uncomment if using command line arguments instead of hardcoded settings
# command line options:
# model, sentence dataset, z_score or raw
#whichmodel = sys.argv[1]
#dataset = sys.argv[2]
#toscale = sys.argv[3]

# hardcoded to replicate analysis reported in paper.
# take out if using command line options instead.
whichmodel = "coco"
dataset = "sick"
toscale = "z_score"

if toscale == "z_score":
    transform = True
elif toscale == "raw":
    transform = False

# load model
if whichmodel == "coco":
    model = task.load("../models/coco-speech.zip")
elif whichmodel == "flickr8k":
    model = task.load("../models/flickr8k-speech.zip")

with open("../data/coco/sick/ratings.p", "rb") as f:
    hr = pickle.load(f)
with open("../data/coco/sick/sentences_a.p", "rb") as f:
    sents_a = pickle.load(f)
with open("../data/coco/sick/sentences_b.p", "rb") as f:
    sents_b = pickle.load(f)

# make avg mfcc list, embeddings list, and an avgs
# activations list for each layer, for the sentences
# in A as well as in B. only keep the averaged versions in memory
# (the full mfcc arrays and activation arrays only
# exist within the batch loop)
avg_mfcc_A = []
avg_mfcc_B = []

avg_l0_A = []
avg_l0_B = []
avg_l1_A = []
avg_l1_B = []
avg_l2_A = []
avg_l2_B = []
avg_l3_A = []
avg_l3_B = []
# for coco model
if whichmodel == "coco":
    avg_l4_A = []
    avg_l4_B = []

embeddings_A = []
embeddings_B = []

# process spoken sentences in batches of 100 (which is how they are stored) 
for b in range((len(sents_a)/100)+1):
    print b,
    # use this when using stored mfcc features
    mfcc_a, mfcc_b = load_batch_mfccs(b) 
    # uncomment next line when extracting mfccs from generated speech
    #mfcc_a, mfcc_b = get_batch_mfccs(b)

    # add accelleration for f8k model
    if whichmodel == "flickr8k":
        mfcc_a = tts.add_accel(mfcc_a)
        mfcc_b = tts.add_accel(mfcc_b)
    avg_mfcc_A.extend([m.mean(axis=0) for m in mfcc_a])
    avg_mfcc_B.extend([m.mean(axis=0) for m in mfcc_b])
    embeddings_A.extend(audiovis.encode_sentences(model, mfcc_a))
    embeddings_B.extend(audiovis.encode_sentences(model, mfcc_b))
    states_a = audiovis.layer_states(model, mfcc_a)
    states_b = audiovis.layer_states(model, mfcc_b)
    avg_l0_A.extend([item[:,0,:].mean(axis=0) for item in states_a])
    avg_l0_B.extend([item[:,0,:].mean(axis=0) for item in states_b])
    avg_l1_A.extend([item[:,1,:].mean(axis=0) for item in states_a])
    avg_l1_B.extend([item[:,1,:].mean(axis=0) for item in states_b])
    avg_l2_A.extend([item[:,2,:].mean(axis=0) for item in states_a])
    avg_l2_B.extend([item[:,2,:].mean(axis=0) for item in states_b])
    avg_l3_A.extend([item[:,3,:].mean(axis=0) for item in states_a])
    avg_l3_B.extend([item[:,3,:].mean(axis=0) for item in states_b])
    if whichmodel == "coco":
        avg_l4_A.extend([item[:,4,:].mean(axis=0) for item in states_a])
        avg_l4_B.extend([item[:,4,:].mean(axis=0) for item in states_b])

avg_sims = {}
print "\ndata loaded and fed to model"       
#### calculate all the cosine similarities
MFCC_sims = cosine_similarities(avg_mfcc_A, avg_mfcc_B, transform)
avg_sims[0] = cosine_similarities(avg_l0_A, avg_l0_B, transform)
avg_sims[1] = cosine_similarities(avg_l1_A, avg_l1_B, transform)
avg_sims[2] = cosine_similarities(avg_l2_A, avg_l2_B, transform)
avg_sims[3] = cosine_similarities(avg_l3_A, avg_l3_B, transform)
if whichmodel == "coco":
    avg_sims[4] = cosine_similarities(avg_l4_A, avg_l4_B, transform)
embedding_sims = cosine_similarities(embeddings_A, embeddings_B, transform)

#### WORD MODEL
# load wordmodel
wordmodel = task.load("../models/coco-text.zip")
# tokenize sentences
words_a = [tokenize(s) for s in sents_a]
words_b = [tokenize(s) for s in sents_b]
# calculate cosine distance between embeddings
# (z_scored if transform == True)
emb_a = vis2.encode_sentences(wordmodel, words_a)
emb_b = vis2.encode_sentences(wordmodel, words_b)
wordmodel_embedding_sims = cosine_similarities(emb_a, emb_b, transform)

# calculate edit distance orthographic sentences
edit_distance = [levenshtein(sents_a[i], sents_b[i]) for i in range(len(sents_a))]
norm_edit_distance = [norm_levenshtein(sents_a[i], sents_b[i]) for i in range(len(sents_a))]

# dump as csv file for further processing in R
if whichmodel == "coco":
    with open("{}_{}_{}.csv".format(toscale, whichmodel, dataset), "wb") as f:
        writer = csv.writer(f, delimiter = "\t")
        #write column headers
        writer.writerow(["a", "b",
                    "mfccs_cossim", "0_cossim", "1_cossim",
                    "2_cossim", "3_cossim", "4_cossim", "emb_cossim",
                     "hr", "wordmodel_cossim", "edit_distance", "norm_edit_distance"])
        for i in range(len(sents_a)):
            writer.writerow([sents_a[i], sents_b[i],
                        MFCC_sims[i], avg_sims[0][i], avg_sims[1][i],
                        avg_sims[2][i], avg_sims[3][i], avg_sims[4][i],
                        embedding_sims[i], hr[i], wordmodel_embedding_sims[i],
                         edit_distance[i], norm_edit_distance[i]])
elif whichmodel == "flickr8k":
    with open("{}_{}_{}.csv".format(toscale, whichmodel, dataset), "wb") as f:
        writer = csv.writer(f, delimiter = "\t")
        #write column headers
        writer.writerow(["a", "b",
                         "mfccs_cossim", "0_cossim", "1_cossim",
                         "2_cossim", "3_cossim", "emb_cossim",
                         "hr", "wordmodel_cossim", "edit_distance", "norm_edit_distance"])
        for i in range(len(sents_a)):
            writer.writerow([sents_a[i], sents_b[i],
                             MFCC_sims[i], avg_sims[0][i], avg_sims[1][i],
                             avg_sims[2][i], avg_sims[3][i], embedding_sims[i],
                             hr[i], wordmodel_embedding_sims[i],
                             edit_distance[i], norm_edit_distance[i]])

if whichmodel == "coco":
    layers = [0, 1, 2, 3, 4]
elif whichmodel == "flickr8k":
    layers = [0, 1, 2, 3]
            
# compute & print correlations
print "Z scored: " + str(transform)
print "\n"

print "Correlations with human ratings: "
print "features\tSpearmans Rho\t\t\t\tPearsons R"
print "MFCC input\t" + str(spearmanr(hr, MFCC_sims)) + "\t" + str(pearsonr(hr, MFCC_sims))
for i in layers:
    print "layer {}:\t".format(i) + str(spearmanr(hr, avg_sims[i])) + "\t" + str(pearsonr(hr, avg_sims[i]))
print "embeddings:\t" + str(spearmanr(hr, embedding_sims)) + "\t" + str(pearsonr(hr, embedding_sims))
print "\n"

print "Correlations with word model: "
print "features\tSpearmans Rho\t\t\t\tPearsons R"
print "MFCC input\t" + str(spearmanr(wordmodel_embedding_sims, MFCC_sims)) + "\t" + str(pearsonr(wordmodel_embedding_sims, MFCC_sims))
for i in layers:
    print "layer {}:\t".format(i) + str(spearmanr(wordmodel_embedding_sims, avg_sims[i])) + "\t" + str(pearsonr(wordmodel_embedding_sims, avg_sims[i]))
print "embeddings:\t" + str(spearmanr(wordmodel_embedding_sims, embedding_sims)) + "\t" + str(pearsonr(wordmodel_embedding_sims, embedding_sims))
print "\n"

print "Correlations with edit distance: "
print "features\tSpearmans Rho\t\t\t\tPearsons R"
print "MFCC input\t" + str(spearmanr(norm_edit_distance, MFCC_sims)) + "\t" + str(pearsonr(norm_edit_distance, MFCC_sims))
for i in layers:
    print "layer {}:\t".format(i) + str(spearmanr(norm_edit_distance, avg_sims[i])) + "\t" + str(pearsonr(norm_edit_distance, avg_sims[i]))
print "embeddings:\t" + str(spearmanr(norm_edit_distance, embedding_sims)) + "\t" + str(pearsonr(norm_edit_distance, embedding_sims))
