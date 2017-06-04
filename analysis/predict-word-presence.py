from time import gmtime, strftime
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD

import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import sys
import random
import imaginet.vendrov_provider as vdp
import imaginet.data_provider as dp

removeStopWords = False
item_count = -1
if len(sys.argv) > 1:
    item_count = int(sys.argv[1])



def applyNeuralNetwork(train_x, train_y, test_x, test_y, label):
    #print "input shape", train_x.shape

    model = Sequential()

    input_size = len(train_x[0])
    model.add(Dense(1024, input_dim=input_size, init='orthogonal', activation='tanh'))
    model.add(Dense(1, init='orthogonal', activation='sigmoid'))

    # Use ADAM optimizer, setting some extra options
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    for j in range(10):
        model.fit(train_x, train_y, nb_epoch=(j+1)*10, batch_size=64, verbose=0)
        trainprd = (numpy.ndarray.flatten(model.predict(train_x, verbose=0))>=0.5).astype('float32')
        prd = (numpy.ndarray.flatten(model.predict(test_x, verbose=0))>=0.5).astype('float32')
    
        print
        print(label + " epoch " + str((j+1)*10))
        print("\tAccuracy on training set: %0.3f" % numpy.mean(trainprd==train_y))
        print("\tAccuracy on test set: %0.3f" % numpy.mean(prd==test_y))
        print
        sys.stdout.flush()
    
    

    pset = set()
    nset = set()
    return(pset,nset)
        

def readStopWords():
    stop_words = set()
    inf = open("stopwords", 'r')
    sw = inf.readline()
    while (sw != ""):
        stop_words.add(sw.strip().lower())
        sw = inf.readline()
    inf.close()
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    return stop_words
                                            

def stimuli(features):
    x = []
    for i in range(len(validate)):
        x += [numpy.concatenate((features[i],embeddings_pos[i]),axis=0), numpy.concatenate((features[i],embeddings_neg[i]),axis=0)]
    return numpy.array(x, dtype='float32')
                    


#for dataset in ['flickr8k','coco']:
for dataset in ['coco']:
    savedir = "../data/%s/"%dataset

    print "load the model and the validation dataset..."
    if dataset == 'flickr8k':
        model = task.load("../models/flickr8k-speech.zip")
        prov = dp.getDataProvider('flickr8k', root='..', audio_kind='human.max1K.accel3.ord.mfcc')
    else:
        model = task.load("../models/coco-speech.zip")
        prov = vdp.getDataProvider(dataset='coco', root='..', audio_kind='mfcc')

    validate = list(prov.iterSentences(split='val'))
    data = [ numpy.asarray(sent['audio'], dtype='float32') for sent in validate ]
    val_embeddings = audiovis.encode_sentences(model, data)
    audiovis = reload(audiovis)
    val_states = audiovis.layer_states(model, data)

    if item_count > -1:
        validate = validate[:min(item_count,len(validate))]
        val_embeddings = val_embeddings[:min(item_count,len(val_embeddings))]
        val_states = val_states[:min(item_count,len(val_states))]

    #split data into training and test
    sp = 2*len(val_embeddings)*4/5
    print "Train: 1-%d; Test: %d-%d\n"%(sp,sp+1,2*len(val_embeddings))
    

    ###predict the presence or absence of a word
    
    # For each sentence, pick a random word as the postive example.
    # Pick a positive example of another sentence as the negative example of the current sentence.
    print "generate positive and negative examples..."
    numpy.random.seed(0)
    random.seed(0)
    stopwords = readStopWords()

    positive = []
    for i in range(len(validate)):
        positem = random.choice(validate[i]['tokens'])
        if (removeStopWords):
            while (positem.lower() in stopwords):
                positem = random.choice(validate[i]['tokens'])
        positive += [positem]

    negative = []
    pmax = len(validate)
    for i in range(pmax):
        negitem = positive[pmax-i-1]
        while (negitem in validate[i]['tokens']):
            negitem = random.choice(positive)
        negative += [negitem]


    #read synthetic representations of word forms from a file

    print "loading audio features..."

    if dataset == 'flickr8k':
        words=numpy.load(savedir+"words-flickr8k.npy")
        audiofeatures = numpy.load(savedir+"mfcc-flickr8k.npy")
    else:
        words=numpy.load(savedir+"words-coco.npy")
        audiofeatures = numpy.load(savedir+"mfcc-coco.npy")

    lexicon = dict(zip(words, audiofeatures))
    mfcc_pos = [lexicon[w] for w in positive]
    mfcc_neg = [lexicon[w] for w in negative]

    embeddings_pos = audiovis.encode_sentences(model, [ numpy.asarray(x, dtype='float32') for x in mfcc_pos ])
    embeddings_neg = audiovis.encode_sentences(model, [ numpy.asarray(x, dtype='float32') for x in mfcc_neg ])
    

    pwrong = {}
    nwrong = {}

    #Predict the presence of a word in a sentence using a neural network
    y = numpy.array([1,0] * len(validate), dtype='float32')

    label = "\nSentence embeddings"
    x = stimuli(val_embeddings)
    (pwrong['embeddings'], nwrong['embeddings']) = applyNeuralNetwork(x[0:sp], y[0:sp], x[sp:], y[sp:], label)
    print


    label = "\nAverage input vectors"
    x = stimuli([numpy.average(item['audio'],axis=0) for item in validate])
    (pwrong['input'], nwrong['input']) = applyNeuralNetwork(x[0:sp], y[0:sp], x[sp:], y[sp:], label)
    print

    layers = val_states[0].shape[1]
    
    label = "\nAverage activation units"
    for l in range(layers):
        x = stimuli([item[:,l,:].mean(axis=0) for item in val_states])
        (pwrong['avg'+str(l)], nwrong['avg'+str(l)]) = applyNeuralNetwork(x[0:sp], y[0:sp], x[sp:], y[sp:], label+"\ton layer %d"%l)
    print
    print

    label = "\nAverage normalized activation units"
    for l in range(layers):
        x = stimuli([item[:,l,:].mean(axis=0) for item in val_states])
        (pwrong['l2avg'+str(l)], nwrong['l2avg'+str(l)]) = applyNeuralNetwork(x[0:sp], y[0:sp], x[sp:], y[sp:], label+"\ton layer %d"%l)
    print
    print

    label ="\nActivation units at the last time step"
    for l in range(layers):
        x = stimuli([item[-1][l] for item in val_states])
        (pwrong['last'+str(l)], nwrong['last'+str(l)]) = applyNeuralNetwork(x[0:sp], y[0:sp], x[sp:], y[sp:], label+"\ton layer %d"%l)
    print
    print
        
