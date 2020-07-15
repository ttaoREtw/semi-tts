import torch
import random
import numpy as np
from functools import partial
from src.text import load_text_encoder
from src.audio import load_audio_transform
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


SPEC_PAD_VALUE = 0 # Spectrogram was in log-scale

def load_dataset(n_jobs, use_gpu, pin_memory, corpus, audio, inference_stage=False):
    ''' Prepare audio dataloader for solver '''
    test_set = None
    # Meta-data related
    data_msg = []
    ### Audio converter (for all kinds of transform/inverse-transform)
    audio_converter = load_audio_transform(**audio)
    data_msg.append('Audio spec.| Feature type = {}\t\t| Feature dim = {}'\
                    .format(audio_converter.feat_type,audio_converter.feat_dim))
    ### Text loader (if exist, return ground truth phone sequence)
    tokenizer = load_text_encoder('phoneme', vocab_file=corpus['vocab_file'], map_table=corpus['map_table'])
    data_msg.append('Text spec. | Token type = {}\t| Vocab size = {}'\
                    .format(tokenizer.token_type,tokenizer.vocab_size))

    # Date related
    ### Load all dataset
    unpair_set, pair_set, dev_set, test_set, set_msg = create_dataset( **corpus, inference_stage=inference_stage)
    data_msg.extend(set_msg)
    ### Create dataloader
    tr_collect = partial(collect_fn, audio_converter=audio_converter, text_loader=tokenizer, mode='train')
    dv_collect = partial(collect_fn, audio_converter=audio_converter, text_loader=tokenizer, mode='dev')

    unpair_set = DataLoader(unpair_set,
                            batch_size=unpair_set.bs_for_collate,
                            shuffle= not inference_stage,
                            drop_last= not inference_stage,
                            collate_fn= dv_collect if inference_stage else tr_collect,
                            num_workers=max(0,n_jobs),
                            pin_memory=pin_memory,
                            worker_init_fn=_worker_init
                           )

    pair_set = DataLoader(pair_set,
                          batch_size=pair_set.bs_for_collate,
                          shuffle= not inference_stage,
                          drop_last= not inference_stage,
                          collate_fn= dv_collect if inference_stage else tr_collect,
                          num_workers=max(0,n_jobs),
                          pin_memory=pin_memory,
                          worker_init_fn=_worker_init
                         )

    dev_set = DataLoader(dev_set,
                         batch_size=dev_set.bs_for_collate,
                         shuffle=False,
                         drop_last=False,
                         collate_fn=dv_collect,
                         num_workers=max(0,n_jobs),
                         pin_memory=pin_memory,
                         worker_init_fn=_worker_init
                        )
    
    if inference_stage:
        test_set = DataLoader(test_set,
                              batch_size=test_set.bs_for_collate,
                              shuffle=False,
                              drop_last=False,
                              collate_fn=dv_collect,
                              num_workers=max(0,n_jobs),
                              pin_memory=pin_memory,
                              worker_init_fn=_worker_init
                             )
    # Augmentation
    data_msg.append('Augment    | Speed rate = {}\t| S/N rate = {}'\
            .format( audio_converter.time_stretch_range, audio_converter.snr_range))

    return unpair_set, pair_set, dev_set, test_set, audio_converter, tokenizer, data_msg
    

def create_dataset(name, path, bucketing, batch_size, spkr_map, partition_table, inference_stage, **kwargs):
    ''' Interface for creating all kinds of dataset'''
    
    msg_list = []

    # Identify corpus
    if name.lower() == "vctk":
        from corpus.vctk import VCTKDataset as Dataset
    else:
        raise NotImplementedError

    # Messages to show
    # Create dataset (ToDo: seperate text-only, speech-only set)
    msg_list = _data_msg(name, path, batch_size)

    pair_set = Dataset(path, partition_table, 'paired', bucketing, batch_size, spkr_map)
    unpair_set = Dataset(path, partition_table, 'unpaired', bucketing, batch_size, spkr_map)
    dev_set = Dataset(path, partition_table, 'dev', bucketing, batch_size, spkr_map)
    test_set = None
    if inference_stage:
        test_set = Dataset(path, partition_table, 'test', bucketing, batch_size, spkr_map)
        msg_list.append(test_set.get_statics())
    else:
        msg_list.append(pair_set.get_statics())
        msg_list.append(unpair_set.get_statics())
        msg_list.append(dev_set.get_statics()) 
    
    return unpair_set, pair_set, dev_set, test_set, msg_list


def collect_fn(batch, audio_converter, text_loader, mode):
    '''Collects a batch of data, returning audio/text according to partial function 
       e.g. [(file1 <str>, sid1 <int>), (file2 <str>, sid2 <int>),...] '''
    
    # Bucketed batch should be [[file1, file2,...]]
    mel, aug_mel, linear, text, sid = None, None, None, None, None
    if type(batch[0]) is list:
        batch = batch[0]

    # Flip a coin to inverse seq order
    # inverse = mode=='train' and audio_converter is not None and random.random()<=audio_converter.inverse_prob
    fpath, sid = zip(*batch)
    
    # Load audio features
    # Read batch: [(file_0, sid_0, (mel_0, aug_mel_0, linear_0)), ...]
    feat = [(f, si, audio_converter.wave_to_feat(f)) 
            for f, si in zip(fpath, sid)]
    # Descending order (also sort batch [list of file path])
    fpath, sid, feat = zip(*list(sorted(feat, reverse=True, key=lambda x:len(x[2][0]))))
    # Unpack audio
    mel, aug_mel, linear = zip(*feat)
    # Zero-padding (mel/aug_mel are padded into same length)
    mel = pad_sequence(mel, batch_first=True, padding_value=SPEC_PAD_VALUE)
    aug_mel = pad_sequence(aug_mel, batch_first=True, padding_value=SPEC_PAD_VALUE)
    linear = pad_sequence(linear, batch_first=True, padding_value=SPEC_PAD_VALUE)

    sid = torch.LongTensor(sid)
    # Load text encoding
    text = [torch.LongTensor(text_loader.file_to_seq(f)) for f in fpath]
    # ToDo: half length if using librispeech
    #if (mode=='train') and (len(text[0])>HALF_BATCH_LEN):
    #    # Half batch size according to text len only when audio feature not used
    #    text = text[:len(text)//2]
    text = pad_sequence(text, batch_first=True)
    
    return mel, aug_mel, linear, sid, text


def _worker_init(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)


def _data_msg(name,path,batch_size):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Corpus = {} (from {})\t| Batch size = {}'\
                    .format(name,path,batch_size))
    return msg_list
