import imaginet.tts as tts
import cPickle as pickle
import sys
import numpy as np
import logging
import pydub
import base64

audiodir = "../data/coco/sick/wav/"

def synthesized(text):
    return open("{}/{}.wav".format(audiodir, base64.urlsafe_b64encode(text)), 'rb').read()



def main():
    logging.getLogger().setLevel('INFO')
    for path in ["a", "b"]:
        sentences_written = "../data/coco/sick/sentences_{}".format(path)

        with open("{}.p".format(sentences_written), 'rb') as f:
            logging.info("Loading {}.p".format(sentences_written))
            sentences = pickle.load(f)


        b = 0
        while b < (len(sentences)/100):
            batch = sentences[(b*100):((b+1)*100)]
            spoken = [synthesized(sent) for sent in batch]
            mfcc = [ tts.extract_mfcc(audio) for audio in spoken ]
            dest = "../data/coco/sick/mfccs/{}_{}.npy".format(path, b)
            np.save(dest, mfcc)
            logging.info("Saved batch to {}".format(dest))
            b += 1
        batch = sentences[(b*100):]
        spoken = [synthesized(sent) for sent in batch]
        mfcc = [ tts.extract_mfcc(audio) for audio in spoken ]
        dest = "../data/coco/sick/mfccs/{}_{}.npy".format(path, b)
        np.save(dest, mfcc)
        logging.info("Saved batch to {}".format(dest))

if __name__ == '__main__':
    main()
