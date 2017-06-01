import imaginet.simple_data as sd
import imaginet.data_provider as dp
import imaginet.vendrov_provider as vendrov
import imaginet.experiment as E
import imaginet.defn.audiovis_rhn as Speech
import imaginet.defn.visual2_rhn as Text
import numpy
import sys

def audio(sent):
    return sent['audio']

def scores(data):
     return (numpy.mean(data['recall'][1]), \
                 numpy.mean(data['recall'][5]),\
                 numpy.mean(data['recall'][10]),\
                 numpy.median(data['ranks']))

def main():

    batch_size = 32
    print "model r@1 r@5 r@10 rank"

    # flickr8k speech
    prov = dp.getDataProvider('flickr8k', root='..', audio_kind='human.max1K.accel3.ord.mfcc')
    data = sd.SimpleData(prov, min_df=10, scale=False,
                         batch_size=batch_size, shuffle=True)
    result = E.evaluate(prov, model_path="../models/flickr8k-speech.zip",
                              task=Speech.Visual,
                              encode_sentences=Speech.encode_sentences,
                              tokenize=audio,
                              split='test',
                              batch_size=batch_size)

    print "flick8k-speech {:.3f} {:.3f} {:.3f} {}".format(*scores(result))
    # flickr8k text
    prov = dp.getDataProvider('flickr8k', root='..', audio_kind=None)
    data = sd.SimpleData(prov, min_df=1, scale=False,
                         batch_size=batch_size, shuffle=True, tokenize=sd.words,
                         val_vocab=True)
    result = E.evaluate(prov, model_path="../models/flickr8k-text.zip",
                              task=Text.Visual,
                              encode_sentences=Text.encode_sentences,
                              tokenize=sd.words,
                              split='test',
                              batch_size=batch_size)
    print "flickr8k-text {:.3f} {:.3f} {:.3f} {}".format(*scores(result))

    # coco speech
    prov = vendrov.getDataProvider('coco', root='..', audio_kind='mfcc')
    data = sd.SimpleData(prov, min_df=10, scale=False,
                         batch_size=batch_size, shuffle=True)
    result = E.evaluate(prov, model_path="../models/coco-speech.zip",
                              task=Speech.Visual,
                              encode_sentences=Speech.encode_sentences,
                              tokenize=audio,
                              split='test',
                              batch_size=batch_size)
    print "coco-speech {:.3f} {:.3f} {:.3f} {}".format(*scores(result))

    # coco text
    batch_size = 128
    prov = vendrov.getDataProvider('coco', root='..', audio_kind=None)
    data = sd.SimpleData(prov, min_df=1, scale=False,
                         batch_size=batch_size, shuffle=True, tokenize=sd.words,
                         val_vocab=True)
    result = E.evaluate(prov, model_path="../models/coco-text.zip",
                              task=Text.Visual,
                              encode_sentences=Text.encode_sentences,
                              tokenize=sd.words,
                              split='test',
                              batch_size=batch_size)
    print "coco-text {:.3f} {:.3f} {:.3f} {}".format(*scores(result))

if __name__ == '__main__':
    main()
