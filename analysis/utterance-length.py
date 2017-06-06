import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.data_provider as dp
import imaginet.vendrov_provider as vdp
import sklearn.linear_model as lm
import sys

remove_stopwords = False
item_count = -1
if len(sys.argv) > 1:
    item_count = int(sys.argv[1])
    


def applyLinearRegression(train_x, train_y, test_x, test_y):
    y = numpy.asarray(train_y).reshape((len(train_y),1))
    rgr = lm.LinearRegression()
    rgr.fit(numpy.asarray(train_x),y)
    return rgr.score(test_x, test_y)

def readStopWords():
    stop_words = set()
    inf = open("stopwords", 'r')
    sw = inf.readline()
    while (sw != ""):
        stop_words.add(sw.strip().lower())
        sw = inf.readline()
    inf.close()
    return stop_words

def removeStopWords(words):
    stop_words = readStopWords()
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    return [i.lower() for i in words if i.lower() not in stop_words]

def minall(scores):
    mmin = 1.0
    for x in scores:
        mmin = min(mmin, min(scores[x]))
    return mmin

def maxall(scores):
    mmax = 0.0
    for x in scores:
        mmax = max(mmax, max(scores[x]))
    return mmax
                                                


rsquare = {}
baseline = {}
for dataset in ['flickr8k','coco']:
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
    sp = len(val_embeddings)*4/5
    print "Train: 1-%d; Test: %d-%d\n"%(sp,sp+1,2*len(val_embeddings))

    print "build dataset..."
    if remove_stopwords:
        y = [[len(removeStopWords(item['tokens']))] for item in validate]
    else:
        y = [[len(item['tokens'])] for item in validate]
                    
    print("Words based on time steps")
    x = [[len(item['audio'])] for item in validate]
    baseline[dataset] = applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:])
    
    rsquare[dataset] = []
    
    print("Average input vectors")
    x = [numpy.average(item['audio'],axis=0) for item in validate]
    rsquare[dataset].append(applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:]))
    
    print("Normalized average activation units")
    layers = val_states[0].shape[1]
    for l in range(layers):
        x = [item[:,l,:].mean(axis=0) for item in val_states]
        xnorm = [v/numpy.linalg.norm(v) for v in x]
        rsquare[dataset].append(applyLinearRegression(xnorm[0:sp], y[0:sp], xnorm[sp:], y[sp:]))
        
    print("Words based on embeddings")
    x = val_embeddings
    rsquare[dataset].append(applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:]))


#----plotting
    
from matplotlib import pyplot as plt

clen = len(acc['coco'])
flen = len(acc['flickr8k'])

xaxis = [i for i in range(clen)]

plt.axis([-1,clen,minall(acc)-0.05, maxall(acc)+0.05])
plt.text(clen-1.5, acc['coco'][-1]-0.05, 'embeddings',color='blue')
plt.text(flen-1.5, acc['flickr8k'][-1]-0.05, 'embeddings', color='red')
plt.xlabel("Network layers")
plt.ylabel("Sentence length prediction (R2)")

#plt.plot(xaxis,[baseline['coco']]*len(xaxis), 'b.')

plt.plot(xaxis[0:2],acc['coco'][0:2],'b--')
coco, = plt.plot(xaxis[1:clen-1],acc['coco'][1:clen-1],'b-', label="COCO")
plt.plot(xaxis[clen-2:],acc['coco'][clen-2:],'b--')
plt.plot([clen-1], acc['coco'][-1], 'bo')

plt.plot(xaxis[0:2],acc['flickr8k'][0:2],'r--')
flickr, = plt.plot(xaxis[1:flen-1],acc['flickr8k'][1:flen-1],'r-', label="Flickr8k")
plt.plot(xaxis[flen-2:flen],acc['flickr8k'][flen-2:],'r--')
plt.plot([flen-1], acc['flickr8k'][-1], 'ro')

plt.legend([coco,flickr], ["COCO","Flickr8k"], loc=4)
plt.savefig('sentlength.pdf')
