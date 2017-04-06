# First stab: Like MultitaskLMD, but trained on disjoint data
# - task 1: textual encoder, visual decoder
# - task 2: textual encoder, textual decoder
# Parameters of textual encoder are shared
# Task 2 may involve for example sentence reconstruction

from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  \
                             last, softmax3d, params

import numpy
import funktional.util as util
from funktional.util import autoassign
import funktional.context as context
import theano.tensor as T
import theano
import zipfile
import cStringIO as StringIO
import json
import cPickle as pickle

class Task(Layer):
    """Task is a trainable Layer.

    You need to set the following attributes:
    - inputs - list of theano symbolic variables
    - target - theano symbolic variable
    - updater - optimizer object (e.g. SGD or Adam)
    """
    inputs = None
    target = None
    updater = None


    def cost(self, target, prediction):
        raise NotImplementedError

    def _make_train(self):
        """Compile function for training."""
        with context.context(training=True):
            prediction = self(*self.inputs)
            thecost = self.cost(self.target, prediction)
        return theano.function(self.inputs + [self.target],
                               thecost,
                               updates=self.updater.get_updates(self.params(), thecost))

    def _make_loss_test(self):
        """Compile function for computing the loss function at test time."""
        with context.context(training=False):
            prediction = self(*self.inputs)
            thecost = self.cost(self.target, prediction)
        return theano.function(self.inputs + [self.target], thecost)

    def _make_predict(self):
        """Compile function for computing the target."""
        with context.context(training=False):
                prediction = self(*self.inputs)
        return theano.function(self.inputs, prediction)

    def compile(self):
        """Compiles theano functions and adds them to self."""
        self.train     = self._make_train()
        self.loss_test = self._make_loss_test()
        self.predict   = self._make_predict()


class Bundle():

    """Interface for combinations of task/data."""

    def params(self):
        raise NotImplementedError

    def weights(self):
        return [ param.get_value() for param in self.params() ]

    def get_config(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def save(self, path):
        zf = zipfile.ZipFile(path, 'w')
        buf = StringIO.StringIO()
        numpy.save(buf, self.weights())
        zf.writestr('weights.npy', buf.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('config.json', json.dumps(self.get_config()),
                    compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('data.pkl', pickle.dumps(self.get_data()),
                    compress_type=zipfile.ZIP_DEFLATED)

class GenericBundle(Bundle):
    """Generic subclass of Bundle which stores common types of settings."""
    def __init__(self, data, config, task, weights=None):
        self.config = config
        self.config['task'] = pickle.dumps(task)
        self.data = data
        self.batcher = data['batcher']
        self.scaler = data['scaler']
        if config.get('size_vocab') is None:
            self.config['size_vocab'] = self.data['batcher'].mapper.size()
        else:
            self.config['size_vocab'] = config['size_vocab']
        self.task = task(config)
        if weights is not None:
            assert len(self.task.params())==len(weights)
            for param, weight in zip(self.params(), weights):
                param.set_value(weight)
        self.task.compile()
        self.task.representation = self.task._make_representation()
        self.task.pile = self.task._make_pile()

    def params(self):
        return self.task.params()

    def get_config(self):
        return self.config

    def get_data(self):
        return self.data


# The following functions work on GenericBundle

def load(path):
    """Load data and reconstruct model."""
    with zipfile.ZipFile(path,'r') as zf:
        buf = StringIO.StringIO(zf.read('weights.npy'))
        weights = numpy.load(buf)
        config  = json.loads(zf.read('config.json'))
        data  = pickle.loads(zf.read('data.pkl'))
        task = pickle.loads(config['task'].encode('utf-8'))
    return GenericBundle(data, config, task, weights=weights)

def representation(model, sents, batch_size=128):
    """Project sents to hidden state space using model.

    For each sentence returns a vector corresponding the activation of the hidden layer
    at the end-of-sentence symbol.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.task.representation(model.batcher.batch_inp(batch))[:,-1,:]
                            for batch in util.grouper(inputs, batch_size) ])

def states(model, sents, batch_size=128):
    """Project each symbol in each sentence in sents to hidden state space
    using model.

    For each sentence returns a matrix corresponding to the
    activations of the top hidden layer at each position in the
    sentence.

    """
    return [ r[:,-1,:] for r in pile(model, sents, batch_size=128) ]

def pile(model, sents, batch_size=128):
    """Project each symbol in each sentence in sents to hidden state
    spaces corresponding to layers using model.

    For each sentence returns a 3D tensor corresponding to the
    activations of the hidden layers at each position in the sentence.

    """
    lens = map(len, sents)
    inputs = list(model.batcher.mapper.transform(sents))
    rs = [ r for batch in util.grouper(inputs, batch_size)
               for r in model.task.pile(model.batcher.batch_inp(batch)) ]
    return [ r[-l-1:,:,:] for (r,l) in zip(rs, lens) ]

def encode_sentences(model, sents, batch_size=128):
    """Project sents to the joint space using model.

    For each sentence returns a vector.
    """
    inputs = list(model.batcher.mapper.transform(sents))
    return numpy.vstack([ model.task.predict(model.batcher.batch_inp(batch))
                            for batch in util.grouper(inputs, batch_size) ])

def encode_images(model, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([ model.task.encode_images(batch)
                          for batch in util.grouper(imgs, batch_size) ])
