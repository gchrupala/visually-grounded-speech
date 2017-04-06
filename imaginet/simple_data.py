import numpy
import cPickle as pickle
import gzip
import os
import copy
import funktional.util as util
from funktional.util import autoassign
from  sklearn.preprocessing import StandardScaler
import string
import random
# Types of tokenization

def words(sentence):
    return sentence['tokens']

def characters(sentence):
    return list(sentence['raw'])

def compressed(sentence):
    return [ c.lower() for c in sentence['raw'] if c in string.letters ]

def phonemes(sentence):
    return [ pho for pho in sentence['ipa'] if pho != "*" ]

class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

class InputScaler():

    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        flat = numpy.vstack(data)
        self.scaler.fit(flat)
        return [ self.scaler.transform(X) for X in data ]
    
    def transform(self, data):
        return [ self.scaler.transform(X) for X in data ]
    
    def inverse_transform(self, data):
        return [ self.scaler.inverse_transform(X) for X in data ]
    
def vector_padder(vecs):
        """Pads each vector in vecs with zeros at the beginning. Returns 3D tensor with dimensions:
           (BATCH_SIZE, SAMPLE_LENGTH, NUMBER_FEATURES).
        """
           
        max_len = max(map(len, vecs))
        return numpy.array([ numpy.vstack([numpy.zeros((max_len-len(vec),vec.shape[1])) , vec]) 
                            for vec in vecs ], dtype='float32')
    
class Batcher(object):

    def __init__(self, mapper, pad_end=False):
        autoassign(locals())
        self.BEG = self.mapper.BEG_ID
        self.END = self.mapper.END_ID
        
    def pad(self, xss): # PAD AT BEGINNING
        max_len = max((len(xs) for xs in xss))
        def pad_one(xs):
            if self.pad_end:
                return xs + [ self.END for _ in range(0,(max_len-len(xs))) ] 
            return [ self.BEG for _ in range(0,(max_len-len(xs))) ] + xs
        return [ pad_one(xs) for xs in xss ]

    def batch_inp(self, sents):
        mb = self.padder(sents)
        return mb[:,1:]

    def padder(self, sents):
        return numpy.array(self.pad([[self.BEG]+sent+[self.END] for sent in sents]), dtype='int32')

    
    def batch(self, gr):
        """Prepare minibatch. 
        
        Returns:
        - input string
        - visual target vector
        - output string at t-1
        - target string
        """
        mb_inp = self.padder([x['tokens_in'] for x in gr])
        mb_target_t = self.padder([x['tokens_out'] for x in gr])
        inp = mb_inp[:,1:]
        target_t = mb_target_t[:,1:]
        target_prev_t = mb_target_t[:,0:-1]
        target_v = numpy.array([ x['img'] for x in gr ], dtype='float32')
        audio = vector_padder([ x['audio'] for x in gr ]) if x['audio']  is not None else None
        return { 'input': inp, 
                 'target_v':target_v, 
                 'target_prev_t':target_prev_t, 
                 'target_t':target_t,
                 'audio': audio }

    
class SimpleData(object):
    """Training / validation data prepared to feed to the model."""
    def __init__(self, provider, tokenize=words, min_df=10, scale=True, scale_input=False, batch_size=64, shuffle=False, limit=None, curriculum=False, val_vocab=False):
        autoassign(locals())
        self.data = {}
        self.mapper = util.IdMapper(min_df=self.min_df)
        self.scaler = StandardScaler() if scale else NoScaler()
        self.audio_scaler = InputScaler() if scale_input else NoScaler()

        parts = insideout(self.shuffled(arrange(provider.iterImages(split='train'), 
                                                               tokenize=self.tokenize, 
                                                               limit=limit)))
        parts_val = insideout(self.shuffled(arrange(provider.iterImages(split='val'), tokenize=self.tokenize)))
        # TRAINING
        if self.val_vocab:
            _ = list(self.mapper.fit_transform(parts['tokens_in'] + parts_val['tokens_in']))
            parts['tokens_in'] = self.mapper.transform(parts['tokens_in']) # FIXME UGLY HACK
        else:
            parts['tokens_in'] = self.mapper.fit_transform(parts['tokens_in'])
            
        parts['tokens_out'] = self.mapper.transform(parts['tokens_out'])
        parts['img'] = self.scaler.fit_transform(parts['img'])
        parts['audio'] = self.audio_scaler.fit_transform(parts['audio'])
        self.data['train'] = outsidein(parts)

        # VALIDATION
        parts_val['tokens_in'] = self.mapper.transform(parts_val['tokens_in'])
        parts_val['tokens_out'] = self.mapper.transform(parts_val['tokens_out'])
        parts_val['img'] = self.scaler.transform(parts_val['img'])
        parts_val['audio'] = self.audio_scaler.transform(parts_val['audio'])
        self.data['valid'] = outsidein(parts_val)
        self.batcher = Batcher(self.mapper, pad_end=False)
        
    def shuffled(self, xs):
        if not self.shuffle:
            return xs
        else:
            zs = copy.copy(list(xs))
            random.shuffle(zs)
            return zs
        
 
    def iter_train_batches(self):
        # sort data by length
        if self.curriculum:
            data = [self.data['train'][i] for i in numpy.argsort([len(x['tokens_in']) for x in self.data['train']])]
        else:
            data = self.data['train']
        for bunch in util.grouper(data, self.batch_size*20):
            bunch_sort = [ bunch[i] for i in numpy.argsort([len(x['tokens_in']) for x in bunch]) ]
            for item in util.grouper(bunch_sort, self.batch_size):
                yield self.batcher.batch(item)
        
    def iter_valid_batches(self):
        for bunch in util.grouper(self.data['valid'], self.batch_size*20):
            bunch_sort = [ bunch[i] for i in numpy.argsort([len(x['tokens_in']) for x in bunch]) ]
            for item in util.grouper(bunch_sort, self.batch_size):
                yield self.batcher.batch(item)

                
    def dump(self, model_path):
        """Write scaler and batcher to disc."""
        pickle.dump(self.scaler, gzip.open(os.path.join(model_path, 'scaler.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.batcher, gzip.open(os.path.join(model_path, 'batcher.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

def arrange(data, tokenize=words, limit=None):
    for i,image in enumerate(data):
        if limit is not None and i > limit:
            break
        for sent in image['sentences']:
            toks = tokenize(sent)
            yield {'tokens_in':  toks, 
                   'tokens_out': toks, 
                   'audio':       sent.get('audio'),
                   'img':        image['feat']}
                   
            
def insideout(ds):
    """Transform a list of dictionaries to a dictionary of lists."""
    ds  = list(ds)
    result = dict([(k, []) for k in ds[0].keys()])
    for d in ds:
        for k,v in d.items():
            result[k].append(v)
    return result

def outsidein(d):
    """Transform a dictionary of lists to a list of dictionaries."""
    ds = []
    keys = d.keys()
    for key in keys:
        d[key] = list(d[key])
    for i in  range(len(d.values()[0])):
        ds.append(dict([(k, d[k][i]) for k in keys]))
    return ds

