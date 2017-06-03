import numpy
import imaginet.tts as tts
import imaginet.vendrov_provider as vdp
import imaginet.data_provider as dp
import logging



def main():
    logging.getLogger().setLevel('INFO')
    for dataset in ['flickr8k', 'coco']:
        
        def mfcc(data):
            if dataset == 'coco': 
                return [ tts.extract_mfcc(audio) for audio in data]
            else:
                return tts.add_accel([ tts.extract_mfcc(audio) for audio in data])
            
        logging.info('Generating features for {}'.format(dataset))
        savedir = "../data/%s/"%dataset
        if dataset == 'flickr8k':
            prov = dp.getDataProvider('flickr8k', root='..', audio_kind=None)
        else:
            prov = vdp.getDataProvider(dataset='coco', root='..', audio_kind=None)
        validate = list(prov.iterSentences(split='val'))


        #build lexicon
        words = set()
        for v in validate:
            for w in v['tokens']:
                words.add(w)

        numpy.save(savedir+"words-"+dataset+".npy", list(words))
        mp3dir = "{}/mp3/".format(savedir)
        spoken = [ tts.decodemp3(open("{}/{}.mp3".format(mp3dir, word), 'rb').read())
                   for word in words ]
        audiofeatures = mfcc(spoken)
        numpy.save(savedir+"mfcc-"+dataset+".npy", audiofeatures)

if __name__ == '__main__':
    main()
    
