from __future__ import division
import imaginet.simple_data as sd
import imaginet.data_provider as dp
import imaginet.vendrov_provider as vendrov
import imaginet.experiment as E
import imaginet.defn.audiovis_rhn as Speech
import imaginet.defn.visual2_rhn as Text
import numpy
import sys
import argparse
import logging

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    retrievalp = commands.add_parser('retrieval')
    retrievalp.set_defaults(func=retrieval)
    errorsp = commands.add_parser('errors')
    errorsp.set_defaults(func=errors)
    args = parser.parse_args()
    args.func(args)

def retrieval(args):

    print "model r@1 r@5 r@10 rank"
    print "flick8k-speech {:.3f} {:.3f} {:.3f} {}".format(*scores(flickr8k_speech()))
    print "flickr8k-text {:.3f} {:.3f} {:.3f} {}".format(*scores(flickr8k_text()))
    print "coco-speech {:.3f} {:.3f} {:.3f} {}".format(*scores(coco_speech()))
    print "coco-text {:.3f} {:.3f} {:.3f} {}".format(*scores(coco_text()))

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
