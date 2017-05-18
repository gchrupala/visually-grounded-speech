import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.data_provider as dp
import sklearn.linear_model as lm
import sys


dataset = 'coco'
remove_stopwords = True
savedir = "/home/aalishah/reimaginet/data/analysis/"

item_count = -1
if len(sys.argv) > 1:
    item_count = int(sys.argv[1])
    


def applyLinearRegression(train_x, train_y, test_x, test_y, label):
    rgr = lm.LinearRegression()
    rgr.fit(train_x, train_y)
    print(label)
    print("\tMean absolute error on test set: %.6f" % numpy.mean(abs(rgr.predict(test_x)-test_y))) 
    print("\tVariation score: %.6f" % rgr.score(test_x, test_y))

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


#load the clean model
print "load the model..."
if dataset == 'flickr':
    model = task.load("/home/gchrupala/reimaginet/examples/audioviz/human-mfcc-rhn-flickr8k.zip")
else:
    model = task.load("/home/gchrupala/reimaginet/examples/audioviz/rhn-mfcc-coco.zip")

print "load the validation dataset..."

if dataset == 'flickr':
    validate=numpy.load(savedir+"validate-flickr.npy")
    val_embeddings=numpy.load(savedir+"embeddings-flickr.npy")
    val_states=numpy.load(savedir+"states-flickr.npy")
else:
    validate=numpy.load(savedir+"validate-coco.npy")
    val_embeddings=numpy.load(savedir+"embeddings-coco.npy")
    val_states=numpy.load(savedir+"states-coco.npy")

if item_count > -1:
        validate = validate[:min(item_count,len(validate))]
        val_embeddings = val_embeddings[:min(item_count,len(val_embeddings))]
        val_states = val_states[:min(item_count,len(val_states))]

print "model and data loaded."
        

###predicting sentence length (number of words)
sp = len(val_embeddings)*4/5
print "Train: 1-%d; Test: %d-%d\n"%(sp,sp+1,len(val_embeddings))

print "build dataset..."
if remove_stopwords:
    y = [len(removeStopWords(item['tokens'])) for item in validate]
else:
    y = [len(item['tokens']) for item in validate]

label = "\nWords based on time steps"
x = [[len(item['audio'])] for item in validate]
applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:], label)
sys.stdout.flush()

label = "\nWords based on embeddings"
x = val_embeddings
applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:], label)
sys.stdout.flush()

label = "\nAverage input vectors"
x = [numpy.average(item['audio'],axis=0) for item in validate]
applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:], label)
sys.stdout.flush()

label = "\nAverage activation units"
for l in range(5):
    x = [item[:,l,:].mean(axis=0) for item in val_states]
    applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:], label+" on layer %d"%l)
    sys.stdout.flush()

label = "\nNormalized average activation units"
for l in range(5):
    x = [item[:,l,:].mean(axis=0) for item in val_states]
    xnorm = [v/numpy.linalg.norm(v) for v in x]
    applyLinearRegression(xnorm[0:sp], y[0:sp], xnorm[sp:], y[sp:], label+" on layer %d"%l)
    sys.stdout.flush()

label = "\nActivation units on the last time step"
for l in range(5):
    x = [item[-1][l] for item in val_states]
    applyLinearRegression(x[0:sp], y[0:sp], x[sp:], y[sp:], label+" on layer %d"%l)
    sys.stdout.flush()

