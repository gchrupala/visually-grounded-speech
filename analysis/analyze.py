from __future__ import division
from __future__ import print_function
import imaginet.simple_data as sd
import imaginet.data_provider as dp
import imaginet.vendrov_provider as vendrov
import imaginet.experiment as E
import imaginet.defn.audiovis_rhn as Speech
import imaginet.defn.visual2_rhn as Text
import imaginet.task
import numpy
import sys
import argparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    retrievalp = commands.add_parser('retrieval')
    retrievalp.set_defaults(func=retrieval)
    errorsp = commands.add_parser('errors')
    errorsp.set_defaults(func=errors)
    homonymsp = commands.add_parser('homonyms')
    homonymsp.set_defaults(func=homonyms)
    args = parser.parse_args()
    args.func(args)

def retrieval(args):

    print("model r@1 r@5 r@10 rank")
    print("flick8k-speech {:.3f} {:.3f} {:.3f} {}".format(*scores(flickr8k_speech())))
    print("flickr8k-text {:.3f} {:.3f} {:.3f} {}".format(*scores(flickr8k_text())))
    print("coco-speech {:.3f} {:.3f} {:.3f} {}".format(*scores(coco_speech())))
    print("coco-text {:.3f} {:.3f} {:.3f} {}".format(*scores(coco_text())))

def errors(args):
    import pandas as pd
    import json
    prov = vendrov.getDataProvider(dataset='coco', root='..', audio_kind=None)
    sent = list(prov.iterSentences(split='val'))

    def extreme(good, worse):
        ratio = numpy.array(good['ranks']) / numpy.array(worse['ranks'])
        return numpy.argsort(ratio)
    def extreme_stats(good, worse, N=100):
        J = extreme(good, worse)[:N]
        L = [len(sent[j]['tokens']) for j in J]
        R = [good['ranks'][j] / worse['ranks'][j] for j in J]
        return (L, R)
    logging.info("Computing scores on validation data")
    score_w = coco_text(split='val')
    score_s = coco_speech(split='val')
    Lw,Rw = extreme_stats(score_w, score_s)
    Ls,Rs = extreme_stats(score_s, score_w)
    data = pd.DataFrame(dict(Length=numpy.hstack([Lw,Ls]),
                             better=numpy.hstack([numpy.repeat("text",100),
                                                  numpy.repeat("speech",100)])))
    logging.info("Writing results to error-length.txt")
    with open("error-length.txt","w") as f:
        f.write(data.to_csv(index=False))

def homonyms(args):

    logging.info("Loading data")
    homonym = [ line.split() for line in open("../data/coco/homonym.txt")]
    prov = vendrov.getDataProvider(dataset='coco', root='..', audio_kind='mfcc')
    sent = list(prov.iterSentences(split='train')) + list(prov.iterSentences(split='val'))
    logging.info("Loading model")
    model = imaginet.task.load("../models/coco-speech.zip")
    def input_mfcc(sent):
        return [ sent_i['audio'].mean(axis=0) for sent_i in sent ]
    def embed(sent):
        return Speech.encode_sentences(model, [ sent_i['audio'] for sent_i in sent ])
    logging.info("Testing on I/O layers")
    with open("ambigu-io.txt", "w") as out:
        print("word1 word2 io count1 count2 majority acc", file=out)
        for H in homonym:
            logging.info("Testing homonym {}".format(H))
            r = test_homonym(H, sent, input_mfcc)
            for acc in r['kfold_acc']:
                print(" ".join(H), "input", r['word1_count'], r['word2_count'], r['majority'], acc, file=out)
            r = test_homonym(H, sent, embed)
            for acc in r['kfold_acc']:
                print(" ".join(H), "output", r['word1_count'], r['word2_count'], r['majority'], acc, file=out)
            out.flush()
    logging.info("Written results to ambigu-io.txt")
    logging.info("Testing on recurrent layers")
    with open("ambigu-layerwise.txt", "w") as out:
        print("word1 word2 layer count1 count2 majority acc", file=out)
        for H in homonym:
            logging.info("Testing homonym {}".format(H))
            for layer in range(5):
                feat = lambda x: mean_layer(x, model, layer=layer)
                r = test_homonym(H, sent, feat)
                for acc in r['kfold_acc']:
                    print(" ".join(H), layer, r['word1_count'], r['word2_count'], r['majority'], acc, file=out)
                    out.flush()
    logging.info("Written results to ambigu-layerwise.txt")

def matching(sent, word):
    for sent_i in sent:
        if word in sent_i['tokens']:
            yield sent_i

def test_homonym(H, sent, features, C=1.0):
    X_0 = features(matching(sent, H[0]))
    X_1 = features(matching(sent, H[1]))
    y_0 = numpy.zeros(len(X_0))
    y_1 = numpy.ones(len(X_1))
    X = normalize(numpy.vstack([X_0, X_1]), norm='l2')
    y = numpy.hstack([y_0, y_1])
    classifier = LogisticRegression(C=C)
    fold = StratifiedKFold(y, n_folds=10)
    score = []
    count = []
    for tr, te in fold:
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        classifier.fit(X_tr, y_tr)
        score.append(sum(classifier.predict(X_te) == y_te))
        count.append(len(y_te))
    score = numpy.array(score, dtype='float')
    count = numpy.array(count, dtype='float')
    result = {'word1_count': len(y_0),
              'word2_count': len(y_1),
              'majority': 1.0 * max(len(y_0),len(y_1))/len(y),
              'kfold_acc': score/count }
    return result

CACHE = {}
def mean_layer(sent, model, layer=0):
    sent = list(sent)
    if len(CACHE) > 5:
        CACHE.clear()
    key = '\n'.join([ sent_i['raw'] for sent_i in sent ])
    if key in CACHE:
        return [ datum[:,layer,:].mean(axis=0) for datum in CACHE[key] ]
    else:
        data = Speech.layer_states(model, [ sent_i['audio'] for sent_i in sent ])
        CACHE[key] = data
        result = [ datum[:,layer,:].mean(axis=0) for datum in data ]
        return result


def flickr8k_speech(split='test'):
    batch_size = 32
    prov = dp.getDataProvider('flickr8k', root='..', audio_kind='human.max1K.accel3.ord.mfcc')
    data = sd.SimpleData(prov, min_df=10, scale=False,
                         batch_size=batch_size, shuffle=True)
    result = E.evaluate(prov, model_path="../models/flickr8k-speech.zip",
                              task=Speech.Visual,
                              encode_sentences=Speech.encode_sentences,
                              tokenize=audio,
                              split=split,
                              batch_size=batch_size)
    return result

def flickr8k_text(split='test'):
    batch_size = 32
    prov = dp.getDataProvider('flickr8k', root='..', audio_kind=None)
    data = sd.SimpleData(prov, min_df=1, scale=False,
                         batch_size=batch_size, shuffle=True, tokenize=sd.words,
                         val_vocab=True)
    result = E.evaluate(prov, model_path="../models/flickr8k-text.zip",
                              task=Text.Visual,
                              encode_sentences=Text.encode_sentences,
                              tokenize=sd.words,
                              split=split,
                              batch_size=batch_size)
    return result

def coco_speech(split='test'):
    batch_size = 32
    prov = vendrov.getDataProvider('coco', root='..', audio_kind='mfcc')
    data = sd.SimpleData(prov, min_df=10, scale=False,
                         batch_size=batch_size, shuffle=True)
    result = E.evaluate(prov, model_path="../models/coco-speech.zip",
                              task=Speech.Visual,
                              encode_sentences=Speech.encode_sentences,
                              tokenize=audio,
                              split=split,
                              batch_size=batch_size)
    return result

def coco_text(split='test'):
    batch_size = 128
    prov = vendrov.getDataProvider('coco', root='..', audio_kind=None)
    data = sd.SimpleData(prov, min_df=1, scale=False,
                         batch_size=batch_size, shuffle=True, tokenize=sd.words,
                         val_vocab=True)
    result = E.evaluate(prov, model_path="../models/coco-text.zip",
                              task=Text.Visual,
                              encode_sentences=Text.encode_sentences,
                              tokenize=sd.words,
                              split=split,
                              batch_size=batch_size)
    return result


def audio(sent):
    return sent['audio']

def scores(data):
     return (numpy.mean(data['recall'][1]), \
                 numpy.mean(data['recall'][5]),\
                 numpy.mean(data['recall'][10]),\
                 numpy.median(data['ranks']))

if __name__ == '__main__':
    main()
