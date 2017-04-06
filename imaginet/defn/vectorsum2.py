from funktional.layer import Layer, Dense, Sum, \
                             Embedding, OneHot,  CosineDistance,\
                             last, softmax3d, params
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

class Encoder(Layer):

    def __init__(self, size_vocab, size_embed):
        autoassign(locals())
        self.Embed = Embedding(self.size_vocab, self.size_embed)
        self.Sum = Sum(self.size_embed)
        
    def params(self):
        return params(self.Embed, self.Sum)
    
    def __call__(self, input):
        return self.Sum(self.Embed(input))

class VectorSum(task.Task):

    def __init__(self, config):
        autoassign(locals())
        self.margin_size = config.get('margin_size', 0.2)
        self.updater = util.Adam(max_norm=config['max_norm'], lr=config['lr'])
        self.Encode = Encoder(config['size_vocab'],
                              config['size_embed'])
        self.ImgEncoder  = Dense(config['size_target'], config['size_embed'], init=eval(config.get('init_img', 'orthogonal')))
        self.inputs = [T.imatrix()]
        self.target = T.fmatrix()
        self.config['margin'] = self.config.get('margin', False)
 
    def compile(self):
        task.Task.compile(self)
        self.encode_images = self._make_encode_images()       
        
    def params(self):
        return params(self.Encode, self.ImgEncoder)
    
    def __call__(self, input):
        # Using last because Sum returns the whole seq of partial sums
        # to be compatible with recurrent layers.
        return util.l2norm(last(self.Encode(input))) 
    
    
    def cost(self, i, s_encoded):
        if self.config['contrastive']:
            i_encoded = util.l2norm(self.ImgEncoder(i))
            return self.contrastive(i_encoded, s_encoded, margin=self.margin_size)
        else:
            raise NotImplementedError
            
    def contrastive(self, i, s, margin=0.2): 
        # i: (fixed) image embedding, 
        # s: sentence embedding
        errors = - util.cosine_matrix(i, s)
        diagonal = errors.diagonal()
        # compare every diagonal score to scores in its column (all contrastive images for each sentence)
        cost_s = T.maximum(0, margin - errors + diagonal)  
        # all contrastive sentences for each image
        cost_i = T.maximum(0, margin - errors + diagonal.reshape((-1, 1)))  
        cost_tot = cost_s + cost_i
        # clear diagonals
        cost_tot = fill_diagonal(cost_tot, 0)

        return cost_tot.mean()
    
   
    def args(self, item):
        return (item['input'], item['target_v'])

    def _make_representation(self):
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_pile(self):
        # This is the same as _make_representation
        with context.context(training=False):
            rep = self.Encode(*self.inputs)
        return theano.function(self.inputs, rep)

    def _make_encode_images(self):
        images = T.fmatrix()
        with context.context(training=False):
            rep = util.l2norm(self.ImgEncoder(images))
        return theano.function([images], rep)

def embeddings(model):
    return model.task.Encode.Embed.params()[0].get_value()

def symbols(model):
    return model.batcher.mapper.ids.decoder
