import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.data_provider as dp
import sklearn.linear_model as lm
import sys

remove_stopwords = False
item_count = -1
if len(sys.argv) > 1:
    item_count = int(sys.argv[1])
    


def applyLinearRegression(train_x, train_y, test_x, test_y):
    x = numpy.asarray(train_x).reshape((len(train_x),1))
    y = numpy.asarray(train_y).reshape((len(train_y),1))
    #tx = numpy.asarray(test_x)
    #ty = numpy.asarray(test_y).reshape((test_y.shape,1))
    rgr = lm.LinearRegression()
    rgr.fit(numpy.asarray(train_x),y)
    return rgr.score(test_x, test_y)
    #print(label)
    #print("\tMean absolute error on test set: %.6f" % numpy.mean(abs(rgr.predict(test_x)-test_y))) 
    #print("\tVariation score: %.6f" % rgr.score(test_x, test_y))

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
    sp = 2*len(val_embeddings)*4/5
    print "Train: 1-%d; Test: %d-%d\n"%(sp,sp+1,2*len(val_embeddings))

    print "build dataset..."
    if remove_stopwords:
        y = [len(removeStopWords(item['tokens'])) for item in validate]
    else:
        y = [len(item['tokens']) for item in validate]

    rsquare[dataset] = []
    
    print("Words based on time steps")
    x = [[len(item['audio'])] for item in validate]
    rgr = applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:])
    baseline[dataset] = rgr

    print("Average input vectors")
    x = [numpy.average(item['audio'],axis=0) for item in validate]
    rsquare[dataset].append(applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:]))

    print("Normalized average activation units")
    for l in range(5):
        x = [item[:,l,:].mean(axis=0) for item in val_states]
        xnorm = [v/numpy.linalg.norm(v) for v in x]
        rsquare[dataset].append(applyLinearRegression(xnorm[0:sp], y[0:sp], xnorm[sp:], y[sp:]))

    print("Words based on embeddings")
    x = val_embeddings
    rsquare[dataset].append(applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:]))

    
import matplotlib.pyplot as plt

xaxis = [0, 1, 2, 3, 4, 5]


plt.axis([-1,6,45,90])
plt.text(3.5, 57, 'embeddings',color='red')
plt.text(4.5, 73, 'embeddings', color='blue')
plt.xlabel("Network layers")
plt.ylabel("Sentence length prediction (R2)")

plt.plot(xaxis[0:2],rsquare['coco'][0:2],'b--')
coco, = plt.plot(xaxis[1:6],rsquare['coco'][1:6],'b-', label="COCO")
plt.plot([5], [75], 'bo')

plt.plot(xaxis[0:2],rsquare['flickr8k'][0:2],'r--')
flickr, = plt.plot(xaxis[1:5],rsquare['flickr8k'][1:5],'r-', label="Flickr8k")
plt.plot([4], [59], 'ro')

plt.legend(handles=[coco,flickr])

plt.savefig('sentence_length.png')
