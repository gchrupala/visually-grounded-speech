import imaginet.simple_data as sd
import imaginet.experiment as E
import imaginet.data_provider as dp
import imaginet.defn.visual2_rhn as D
dataset = 'flickr8k'
batch_size = 32
epochs=20
prov = dp.getDataProvider(dataset, root='/home/gchrupala/repos/reimaginet/', audio_kind=None)
data = sd.SimpleData(prov, min_df=1, scale=False,
                     batch_size=batch_size, shuffle=True, tokenize=sd.words, val_vocab=True)
model_config = dict(size_embed=300, size=1024, depth=1, recur_depth=1, max_norm=2.0, residual=True,
                    drop_i=0.0, drop_s=0.0,
                    lr=0.001, size_target=4096,
                    contrastive=True, margin_size=0.2, fixed=True, 
                    init_img='xavier')
run_config = dict(seed=61, task=D.Visual, epochs=epochs, validate_period=400)



def audio(sent):
    return sent['audio']

eval_config = dict(tokenize=sd.words, split='val', task=D.Visual, batch_size=batch_size,
                   epochs=epochs, encode_sentences=D.encode_sentences)

E.run_train(data, prov, model_config, run_config, eval_config)
#E.run_eval(prov, eval_config, encode_sentences=D.encode_sentences)
