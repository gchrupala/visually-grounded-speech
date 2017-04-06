import imaginet.simple_data as sd
import imaginet.experiment as E
import imaginet.vendrov_provider as dp
import imaginet.defn.audiovis_rhn as D
dataset = 'coco'
batch_size = 32
epochs=15
prov = dp.getDataProvider(dataset, root='/home/gchrupala/repos/reimaginet/', audio_kind='mfcc')
data = sd.SimpleData(prov, min_df=10, scale=False,
                     batch_size=batch_size, shuffle=True)
model_config = dict(size=512, depth=5, recur_depth=2, max_norm=2.0, residual=True,
                    drop_i=0.0, drop_s=0.0,
                    lr=0.0002, size_vocab=13, size_target=4096,
                    filter_length=6, filter_size=64, stride=3,
                    contrastive=True, margin_size=0.2, fixed=True, 
                    init_img='xavier', size_attn=512)
run_config = dict(seed=71, task=D.Visual, epochs=epochs, validate_period=4000)



def audio(sent):
    return sent['audio']

eval_config = dict(tokenize=audio, split='val', task=D.Visual, batch_size=batch_size,
                   epochs=epochs, encode_sentences=D.encode_sentences)

E.run_train(data, prov, model_config, run_config, eval_config, resume=True)
#E.run_eval(prov, eval_config, encode_sentences=D.encode_sentences)
