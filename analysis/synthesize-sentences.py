import imaginet.tts as tts
import cPickle as pickle
import sys

def synthesize(text):
    return tts.decodemp3(tts.speak(text))

# sentences_written is the filename without extension
# of a pickled list of strings
sentences_written = sys.argv[1]

with open("{}.p".format(sentences_written), 'rb') as f:
    sentences = pickle.load(f)

b = 0
while b < (len(sentences)/100):
    batch = sentences[(b*100):((b+1)*100)]
    spoken = [synthesize(sent) for sent in batch]
    pickle.dump(spoken, open("speech/{}_{}.pkl".format(sentences_written, b),"w"), protocol=pickle.HIGHEST_PROTOCOL)
    print b
    b += 1
batch = sentences[(b*100):]
spoken = [synthesize(sent) for sent in batch]
pickle.dump(spoken, open("speech/{}_{}.pkl".format(sentences_written, b),"w"), protocol=pickle.HIGHEST_PROTOCOL)
print b
print "speech generation completed"
