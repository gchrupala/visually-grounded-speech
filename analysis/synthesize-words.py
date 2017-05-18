import numpy
import imaginet.tts as tts
import imaginet.data_provider as dp

savedir = "/home/aalishah/reimaginet/data/analysis/"
dataset = 'coco'
#dataset = 'flickr8k'


if dataset == 'flickr8k':
    prov = dp.getDataProvider('flickr8k', root='/home/gchrupala/reimaginet/', audio_kind='human.max1K.accel3.ord.mfcc')
else:
    prov = dp.getDataProvider('coco', root='/home/gchrupala/reimaginet/', audio_kind='mfcc')
validate = list(prov.iterSentences(split='val'))


#build lexicon
words = set()
for v in validate:
    for w in v['tokens']:
        words.add(w)



def synthesize(text):
    print text
    return tts.decodemp3(tts.speak(text))

    
def speak(data):
    return [synthesize(word)for word in data]

def mfcc(data):
    if dataset == 'coco': 
        return [ tts.extract_mfcc(audio) for audio in data]
    return tts.add_accel([ tts.extract_mfcc(audio) for audio in data])

numpy.save(savedir+"words-"+dataset+".npy", list(words))
spoken = speak(list(words))
numpy.save(savedir+"spoken-"+dataset+".npy", spoken)
audiofeatures = mfcc(spoken)
numpy.save(savedir+"mfcc-"+dataset+".npy", audiofeatures)
