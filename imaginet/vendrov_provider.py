import os
import numpy
import json
import sys
import gzip

class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc', extra_train=True):
    self.root = root
    self.dataset = dataset
    self.audio_kind = audio_kind
    self.img = {}
    self.txt = {}
    self.img['train'] = numpy.load(open("{}/data/{}/vendrov/data/coco/images/10crop/train.npy".format(self.root, self.dataset)))
    self.img['val'] = numpy.load(open("{}/data/{}/vendrov/data/coco/images/10crop/val.npy".format(self.root, self.dataset)))
    self.img['test'] = numpy.load(open("{}/data/{}/vendrov/data/coco/images/10crop/test.npy".format(self.root, self.dataset)))

    self.txt['train'] = [ line.split() for line in open("{}/data/{}/vendrov/data/coco/train.txt".format(self.root, self.dataset)) ]
    self.txt['val'] = [ line.split() for line in open("{}/data/{}/vendrov/data/coco/val.txt".format(self.root, self.dataset)) ]
    self.txt['test'] = [ line.split() for line in open("{}/data/{}/vendrov/data/coco/test.txt".format(self.root, self.dataset)) ]

    audio_path = "{}/data/{}/dataset.{}.npy".format(self.root, self.dataset, self.audio_kind)
    ipa_path   = "{}/data/{}/dataset.ipa.jsonl.gz".format(self.root, self.dataset)
    words = json.load(open("{}/data/{}/dataset.words.json".format(self.root, self.dataset)))
    self.w2i = {}
    for i in range(0, len(words)):
        self.w2i[words[i]] = i
    try:
        self.IPA = [ json.loads(line)['phonemes'] for line in gzip.open(ipa_path) ]
    except IOError as e:
        sys.stderr.write("Warning: could not read file {}: IPA transcription not available\n".format(ipa_path))
        self.IPA = None

    try:

        self.AUDIO = numpy.load(audio_path)
    except IOError as e:
        sys.stderr.write("Warning: could not read file {}: audio features not available\n".format(audio_path))


  def iterImages(self, split='train', shuffle=False):
    ix = range(0, self.img[split].shape[0])
    if shuffle:
      random.shuffle(ix)
    for i in ix:
      img = {}
      img['feat'] = self.img[split][i,:]
      img['sentences'] = []
      img['imgid'] = i
      for j in range(0,5):
        sent = {}
        sent['tokens'] = self.txt[split][i*5+j]
        sent['raw'] = ' '.join(sent['tokens'])
        sent['imgid'] = i
        if self.audio_kind is None:
            sent['audio'] = None
        else:
            sent['audio'] = self.AUDIO[self.w2i[sent['raw']]]
        if self.IPA is not None:
            sent['ipa'] = self.IPA[self.w2i[sent['raw']]]
        img['sentences'].append(sent)
      yield img

  def iterSentences(self, split='train', shuffle=False):
    for img in self.iterImages(split=split, shuffle=shuffle):
      for sent in img['sentences']:
        yield sent


def getDataProvider(*args, **kwargs):
	return Provider(*args, **kwargs)
