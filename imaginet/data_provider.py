# Code adapted from https://github.com/karpathy/neuraltalk
# by Andrej Karpathy

import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
import itertools
import gzip
import sys
import numpy

class BasicDataProvider:
  def __init__(self, dataset, root='.', extra_train=False, audio_kind='fbank'):

    self.root = root
    # !assumptions on folder structure
    self.dataset_root = os.path.join(self.root, 'data', dataset)
    self.image_root = os.path.join(self.root, 'data', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    ipa_path     = os.path.join(self.dataset_root, 'dataset.ipa.jsonl.gz')
    audio_path    = os.path.join(self.dataset_root, 'dataset.{}.npy'.format(audio_kind))
    self.dataset = json.load(open(dataset_path, 'r'))

    # load ipa
    try:
      IPA = {}
      for line in gzip.open(ipa_path):
        item = json.loads(line)
        IPA[item['sentid']] = item['phonemes']
        # add ipa field to dataset
      for image in self.dataset['images']:
        for sentence in image['sentences']:
          sentence['ipa'] = IPA[sentence['sentid']]
    except IOError:
      sys.stderr.write("Warning: could not read file {}: IPA transcription not available\n".format(ipa_path))

    try:
        AUDIO = numpy.load(audio_path)
        sentid = 0
        for image in self.dataset['images']:
            for sentence in image['sentences']:
                sentence['audio'] = AUDIO[sentid]
                sentid += 1
    except IOError:
        sys.stderr.write("Warning: could not read file {}: audio features not available\n".format(audio_path))


    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')

    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['feats']

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      if extra_train and img['split'] == 'restval':
        img['split']='train'
      self.split[img['split']].append(img)

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the
  # data provider class data, but for now lets do the simple thing and
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences':
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    sent = random.choice(img['sentences'])

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]:
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])

def getDataProvider(dataset, root='.', extra_train=False, audio_kind='fbank'):
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['flickr8k', 'flickr30k', 'coco', 'coco+flickr30k'], 'dataset %s unknown' % (dataset, )
  if dataset == 'coco+flickr30k':
    return CombinedDataProvider(datasets=['coco', 'flickr30k'], root=root, extra_train=extra_train, audio_kind=audio_kind)
  else:
    return BasicDataProvider(dataset, root, extra_train=extra_train, audio_kind=audio_kind)

class CombinedDataProvider(object):

  def __init__(self, datasets, root='.', extra_train=False, audio_kind='fbank'):
    self.datasets = datasets
    self.root = root
    self.providers = [ BasicDataProvider(dataset, root=self.root, extra_train=extra_train, audio_kind=audio_kind)
                       for dataset in self.datasets ]

  def getSplitSize(self, split, ofwhat='sentences'):
    return sum((p.getSplitSize(split, ofwhat=ofwhat) for p in self.providers))

  def sampleImageSentencePair(self, split='train'):
    raise NotImplementedError()

  def iterImageSentencesPair(self, split='train', max_images=-1):
    iters = [ p.iterImageSentencePair(split=split, max_images=-1) for p in self.providers ]
    for item in itertools.chain(*iters):
      yield item

  def iterImageSentencePairBatch(self, split='train', max_images=-1, max_batch_size = 100):
    iters = [ p.iterImageSentencePairBatch(split=split, max_images=max_images, max_batch_size=max_batch_size) for p in self.providers ]
    for item in itertools.chain(*iters):
      yield item

  def iterSentences(self, split = 'train'):
    iters = [ p.iterSentences(split=split) for p in self.providers ]
    for item in itertools.chain(*iters):
      yield item

  def iterImages(self, split = 'train', max_images = -1):
    iters = [ p.iterImages(split=split, max_images=max_images) for p in self.providers ]
    for item in itertools.chain(*iters):
      yield item
