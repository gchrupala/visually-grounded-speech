import json
import gzip
from gtts import gTTS
import StringIO
import base64
import pydub
import python_speech_features as features
import StringIO
import scipy.io.wavfile as wav
import numpy
import time
from urllib2 import HTTPError
import funktional.util as util
import numpy.random
import random
import os

def speak(words):
    f = StringIO.StringIO()
    gTTS(text=words, lang='en-us').write_to_fp(f)
    return f.getvalue()

def tryspeak(words,i):
    try:
        return speak(words)
    except:
        print "sleeping {} seconds".format(60*i)
        time.sleep(60*i)
        return tryspeak(words,i*2)
        
def tts(dataset='flickr8k'):
    data = json.load(open('/home/gchrupala/repos/reimaginet/data/{}/dataset.json'.format(dataset)))

    with gzip.open("/home/gchrupala/repos/reimaginet/data/{}/dataset.mp3.jsonl.gz".format(dataset),"w") as f:
        for img in data['images']:
            for s in img['sentences']:
                    audio = tryspeak(s['raw'],1)
                    f.write("{}\n".format(json.dumps({'sentid':s['sentid'], 'speech':base64.b64encode(audio)})))



def decodemp3(s):
    seg = from_mp3(s)
    io = StringIO.StringIO()
    seg.export(io, format='wav')
    return io.getvalue()

def from_mp3(s):
    return pydub.AudioSegment.from_mp3(StringIO.StringIO(s))

# def wavdata(dataset='flickr8k'):
#     with gzip.open("/home/gchrupala/repos/reimaginet/data/flickr8k/dataset.wav.jsonl.gz","w") as out:
#         for line in gzip.open("/home/gchrupala/repos/reimaginet/data/flickr8k/dataset.mp3.jsonl.gz"):
#             sent = json.loads(line)
#             out.write("{}\n".format(json.dumps({'sentid':sent['sentid'],
#                                                 'wav': base64.b64encode(decodemp3(base64.b64decode(sent['speech'])))})))
            

def extract_mfcc(sound):
    (rate,sig) = wav.read(StringIO.StringIO(sound))
    mfcc_feat = features.mfcc(sig,rate)
    return numpy.asarray(mfcc_feat, dtype='float32')

def extract_fbank(sound):
    (rate,sig) = wav.read(StringIO.StringIO(sound))
    fbank_feat = features.logfbank(sig,rate)
    return fbank_feat

def featurefile(dataset='flickr8k', chunksize=1000, kind='fbank', noisy=False):
    if kind == 'mfcc':
        extract = extract_mfcc
    elif kind == 'fbank':
        extract = extract_fbank
    else:
        raise "Invalid kind"
    infix = ".noisy" if noisy else ""
    for i,chunk in enumerate(util.grouper(gzip.open("/home/gchrupala/repos/reimaginet/data/{}/dataset{}.mp3.jsonl.gz".format(dataset, infix)),chunksize)):
        result = []
        for line in chunk:
            sent = json.loads(line)
            sound = decodemp3(base64.b64decode(sent['speech']))
            result.append(extract(sound))
        numpy.save("/home/gchrupala/repos/reimaginet/data/{}/dataset{}.{}.{}.npy".format(dataset,infix,kind,i), result)

import pydub.playback

def noisyspeech(dataset='flickr8k'):
    dir = "/home/gchrupala/repos/reimaginet/data/PCCdata16kHz/train/background/"
    background = sum((pydub.AudioSegment.from_mp3(dir + file) for file in os.listdir(dir)[:10]
                      if file.endswith(".wav")), pydub.AudioSegment.empty())
    print "Background noise loaded"
    with gzip.open("/home/gchrupala/repos/reimaginet/data/{}/dataset.noisy.mp3.jsonl.gz".format(dataset),"w") as out:
        for line in gzip.open("/home/gchrupala/repos/reimaginet/data/{}/dataset.mp3.jsonl.gz".format(dataset)):
            sent = json.loads(line)
            print sent['sentid']
            sound = from_mp3(base64.b64decode(sent['speech']))
#            pydub.playback.play(sound)
            noisy = noisify(sound, background)
#             pydub.playback.play(noisy)
            io = StringIO.StringIO()
            noisy.export(io, format='mp3')
            sent['speech'] = base64.b64encode(io.getvalue())
            out.write(json.dumps(sent))
            out.write("\n")
        
def noisify(sound, noise):
    loudness = random.uniform(0.0, 10.0)
    start = random.randint(0, int((noise.duration_seconds - sound.duration_seconds) * 1000))
    speed = 1 + abs(numpy.random.normal(0.0, 0.1))
    noisy = sound.speedup(playback_speed=speed).overlay(noise[start:] + loudness)
    return noisy

# Delta and acceleration

def delta(v, N=2, offset=1):
    d = numpy.zeros_like(v[:, offset:])
    for t in range(0, d.shape[0]):
        Z = 2 * sum(n**2 for n in range(1, N+1))
        d[t,:] = sum(n * (v[min(t+n, v.shape[0]-1), offset:]-v[max(t-n, 0), offset:]) for n in range(1,N+1)) / Z
    return d

def add_accel(data):
    return numpy.array( [ numpy.hstack([row, delta(row, N=2, offset=1), delta(delta(row, N=2, offset=1), offset=0)]) for row in data ])

def extract_mfcc_accel(sound):
    return add_accel(extract_mfcc(sound))


