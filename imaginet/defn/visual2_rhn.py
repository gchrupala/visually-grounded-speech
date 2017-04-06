from funktional.layer import Layer, Dense, StackedGRU, StackedGRUH0, \
                             Embedding, OneHot,  clipped_rectify, tanh, CosineDistance,\
                             last, softmax3d, params
from funktional.rhn import StackedRHN0
import funktional.context as context        
from funktional.layer import params
import imaginet.task as task
from funktional.util import autoassign
import funktional.util as util
from funktional.util import steeper_sigmoid, sigmoid, orthogonal, xavier
import theano.tensor as T
import theano
import zipfile
import numpy
import StringIO
import json
import cPickle as pickle
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.extra_ops import fill_diagonal

def uniform(shape):
    return util.uniform(shape, scale=0.1)

class Encoder(Layer):


    def __init__(self, size_vocab, size_embed, size, depth=1, recur_depth=1, drop_i=0.75 , drop_s=0.25,
                 residual=False, seed=1):
        autoassign(locals())
        if self.size_embed == 'onehot':
            self.Embed = OneHot(self.size_vocab)
            self.RHN = StackedRHN0(self.size_vocab, self.size, depth=self.depth, recur_depth=self.recur_depth,
                               drop_i=self.drop_i, drop_s=self.drop_s, residual=self.residual, seed=self.seed)
        else:
            self.Embed = Embedding(self.size_vocab, self.size_embed)            
            self.RHN = StackedRHN0(self.size_embed, self.size, depth=self.depth, recur_depth=self.recur_depth,
                               drop_i=self.drop_i, drop_s=self.drop_s, residual=self.residual, seed=self.seed)

    def params(self):
        return params(self.Embed, self.RHN)
    
    def __call__(self, input):
        return self.RHN(self.Embed(input))

class Visual(task.Task):

    def __init__(self, config):
        autoassign(locals())
        self.margin_size = config.get('margin_size', 0.2)
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Encode = Encoder(config['size_vocab'],
                              config['size_embed'], 
                              config['size'],
                              depth=config.get('depth', 1),
                              recur_depth=config.get('recur_depth',1),
                              drop_i=config.get('drop_i', 0.75),
                              drop_s=config.get('drop_s', 0.25),
                              residual=config.get('residual', False),
                              seed=config.get('seed', 1))                              
        self.ImgEncoder  = Dense(config['size_target'], config['size'], init=eval(config.get('init_img', 'orthogonal')))
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()
        
    def compile(self):
        task.Task.compile(self)
        self.encode_images = self._make_encode_images()
        
    def params(self):
        return params(self.Encode, self.ImgEncoder)
    
    def __call__(self, input):
        return util.l2norm(last(self.Encode(input)))

    # FIXME HACK ALERT
    def cost(self, i, s_encoded):
        if self.config['contrastive']:
            i_encoded = util.l2norm(self.ImgEncoder(i))
            return util.contrastive(i_encoded, s_encoded, margin=self.margin_size)
        else:
            raise NotImplementedError
            
    def args(self, item):
        return (item['input'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        with context.context(training=False):
            rep = self.Encode.RHN.intermediate(self.Encode.Embed(*self.inputs))
        return theano.function(self.inputs, rep)

    def _make_encode_images(self):
        images = T.fmatrix()
        with context.context(training=False):
            rep = util.l2norm(self.ImgEncoder(images))
        return theano.function([images], rep)
            
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

def embeddings(model):
    return model.task.Encode.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder
