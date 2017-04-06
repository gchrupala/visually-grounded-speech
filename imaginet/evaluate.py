# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
from __future__ import division
import theano
import theano.tensor as T
import numpy
from scipy.spatial.distance import cdist

def paraphrase_ranking(vectors, group):
    """Rank sentences by projection and return evaluation metrics."""
    return ranking(vectors, vectors, group, ns=[4], exclude_self=True)

def ranking(candidates, vectors, correct, ns=[1,5,10], exclude_self=False):
    """Rank `candidates` in order of similarity for each vector and return evaluation metrics.

    `correct[i][j]` indicates whether for vector i the candidate j is correct.
    """
    #distances = cdist(vectors, candidates, metric='cosine')
    distances = Cdist(batch_size=2**13)(vectors, candidates)
    result = {'ranks' : [] , 'precision' : {}, 'recall' : {}, 'overlap' : {} }
    for n in ns:
        result['precision'][n] = []
        result['recall'][n]    = []
        result['overlap'][n]   = []
    for j, row in enumerate(distances):
        ranked = numpy.argsort(row)
        if exclude_self:
            ranked = ranked[ranked!=j]
        id_correct = numpy.where(correct[j][ranked])[0]
        rank1 = id_correct[0] + 1
        topn = {}
        for n in ns:
            id_topn = ranked[:n]
            overlap = len(set(id_topn).intersection(set(ranked[id_correct])))
            result['precision'][n].append(overlap/n)
            result['recall'   ][n].append(overlap/len(id_correct))
            result['overlap'  ][n].append(overlap)
        result['ranks'].append(rank1)
    return result

class Cdist():
    """Return cosine distances between two sets of vectors."""
    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.U = T.matrix('U')
        self.V = T.matrix('V')
        self.U_norm = self.U / self.U.norm(2, axis=1).reshape((self.U.shape[0], 1))
        self.V_norm = self.V / self.V.norm(2, axis=1).reshape((self.V.shape[0], 1))

        self.W = T.dot(self.U_norm, self.V_norm.T)
        self.cosine = theano.function([self.U, self.V], self.W)

    def __call__(self, A, B):
        if self.batch_size is None:
            chunks = [A]
        else:
            chunks  = numpy.split(A, [i for i
                                      in range(self.batch_size, A.shape[0], self.batch_size) ])
        cosines = numpy.vstack([self.cosine(chunk, B) for chunk in chunks])
        return 1 - cosines
