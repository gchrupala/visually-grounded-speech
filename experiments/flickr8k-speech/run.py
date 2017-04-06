import imaginet.simple_data as sd
import imaginet.experiment as E
import imaginet.data_provider as dp
import imaginet.defn.audiovis_rhn as D
dataset = 'flickr8k'
batch_size = 32
epochs=25
prov = dp.getDataProvider(dataset, root='../..', audio_kind='human.max1K.accel3.ord.mfcc')
data = sd.SimpleData(prov, min_df=10, scale=False,
                     batch_size=batch_size, shuffle=True)
model_config = dict(size=1024, depth=4, recur_depth=2, max_norm=2.0, residual=True,
                    drop_i=0.25, drop_s=0.1,
                    lr=0.0002, size_vocab=37, size_target=4096,
                    filter_length=6, filter_size=64, stride=2,
                    contrastive=True, margin_size=0.2, fixed=True, 
                    init_img='xavier', size_attn=128)
run_config = dict(seed=51, task=D.Visual, epochs=epochs, validate_period=400)



def audio(sent):
    return sent['audio']

eval_config = dict(tokenize=audio, split='val', task=D.Visual, batch_size=batch_size,
                   epochs=epochs, encode_sentences=D.encode_sentences)

E.run_train(data, prov, model_config, run_config, eval_config)

