import math
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
import editdistance as ed
import soundfile as sf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#PRESERVE_INDICES = len(['<pad>', '<space>'])
PRESERVE_INDICES = len(['<pad>', '<space>', '<eos>'])
#IGNORE_INDICES = [0, 1, 41]
IGNORE_INDICES = [0, 1, 2, 42]
SEP = '\t'

class Timer():
    ''' Timer for recording training time distribution. '''
    def __init__(self):
        self.prev_t = time.time()
        self.clear()

    def set(self):
        self.prev_t = time.time()

    def cnt(self,mode):
        self.time_table[mode] += time.time()-self.prev_t
        self.set()
        if mode =='bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg  = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)'.format(**self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd':0,'fw':0,'bw':0}
        self.click = 0

# Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr.py#L168
def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3,4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError
def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias

def freq_loss(pred, label, sample_rate, n_mels, loss, differential_loss, emphasize_linear_low, p=1):
    """
    Args:
        pred: model output
        label: target
        loss: `l1` or `mse`
        differential_loss: use differential loss or not, see here `https://arxiv.org/abs/1909.10302`
        emphasize_linear_low: emphasize the low-freq. part of linear spectrogram or not
        
    Return:
        loss
    """    
    # ToDo : Tao 
    # pred -> BxTxD predicted mel-spec or linear-spec
    # label-> same shape
    # return loss for loss.backward()
    if loss == 'l1':
        criterion = torch.nn.functional.l1_loss
    elif loss == 'mse':
        criterion = torch.nn.functional.mse_loss
    else:
        raise NotImplementedError

    cutoff_freq = 3000

    # Repeat for postnet
    #_, chn, _, dim = pred.shape
    dim = pred.shape[-1]
    #label = label.unsqueeze(1).repeat(1,chn,1,1)

    loss_all = criterion(p * pred, p * label)

    if dim != n_mels and emphasize_linear_low:
        # Linear
        n_priority_freq = int(dim * (cutoff_freq / (sample_rate/2)))
        pred_low = pred[:, :, :n_priority_freq]
        label_low = label[:, :, :n_priority_freq]
        loss_low = criterion(p * pred_low, p * label_low)
        #loss_low = torch.nn.functional.mse_loss(p * pred_low, p * label_low)
        loss_all = 0.5 * loss_all + 0.5 * loss_low

    if dim == n_mels and differential_loss:
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        label_diff = label[:, 1:, :] - label[:, :-1, :]
        loss_all += 0.5 * criterion(p * pred_diff, p * label_diff)

    return loss_all

def feat_to_fig(feat):
    if feat is None:
        return None
    # feat TxD tensor
    data = _save_canvas(feat.numpy().T)
    return torch.FloatTensor(data),"HWC"

def data_to_bar(data, gt_data, tok_size, tick, zero_pad_tok=True):
    if len(gt_data) == 0:
        return None
    # Hack to get discrete bar graph
    cnts = [data.count(i)/len(data) for i in range(tok_size)]
    gt_cnts = [gt_data.count(i)/len(gt_data) for i in range(tok_size)]
    if zero_pad_tok:
        cnts[0] = 0
        gt_cnts[0] = 0
    data = _save_canvas( (cnts,gt_cnts), meta=(range(tok_size),tick))
    return torch.FloatTensor(data),"HWC"

def _save_canvas(data, meta=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    if meta is None:
        ax.imshow(data, aspect="auto", origin="lower")
    else:
        ax.bar(meta[0],data[0],tick_label=meta[1],fc=(0, 0, 1, 0.5))
        ax.bar(meta[0],data[1],tick_label=meta[1],fc=(1, 0, 0, 0.5))
    fig.canvas.draw()
    # Note : torch tb add_image takes color as [0,1]
    data = np.array(fig.canvas.renderer._renderer)[:,:,:-1]/255.0 
    plt.close(fig)
    return data

# Reference : https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])

def cal_per(pred, truth):
    # Calculate error rate of a batch
    if pred is None:
        return np.nan
    elif len(pred.shape)>=3:
        pred = pred.argmax(dim=-1)
    er = []
    for p,t in zip(pred.cpu(),truth.cpu()):
        p = p.tolist()
        p = [v for i,v in enumerate(p) if (i==0 or v!=p[i-1]) and v not in IGNORE_INDICES] # Trim repeat
        t = [v for v in t.tolist() if v not in IGNORE_INDICES]
        er.append(float(ed.eval( p,t))/len(t))
    return sum(er)/len(er)
    

def cal_ppx(prob):
    prob = prob.cpu()
    prob_len = torch.sum(prob.sum(dim=-1)!=0,dim=-1,keepdim=True).float()
    entropy = -torch.sum(prob*(prob+1e-10).log2(),dim=-1) # 2-based log
    entropy = torch.mean(entropy.sum(dim=-1)/prob_len)
    return torch.pow(torch.FloatTensor([2]),entropy)

# Reference : 
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/7e14834dd5e48bb1e6c74581c55684405e821298/transformer/Models.py
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_audio_feat_mask(actual_lengths, n_frames_per_step, dim):
    """
    Return:
        mask with 1 for padded part and 0 for non-padded part
    """
    # padded length = actual lengths + at least 1 frame padded
    padded_lengths = actual_lengths + n_frames_per_step-(actual_lengths%n_frames_per_step)
    max_len = torch.max(padded_lengths).item()
    if max_len % n_frames_per_step != 0:
        max_len += n_frames_per_step - max_len % n_frames_per_step
        assert max_len % n_frames_per_step == 0
    ids = torch.arange(0, max_len).to(actual_lengths.device)
    mask = (ids < padded_lengths.unsqueeze(1)).bool()
    mask = ~mask
    # (D, B, T)
    mask = mask.expand(dim, mask.size(0), mask.size(1))
    # (B, T, D)
    mask = mask.permute(1, 2, 0)
    return mask

def get_seq_mask(lens, max_len=None):
    ''' Mask for given sequence, return shape [B,T,1]'''
    batch_size = len(lens)
    max_l = lens.max() if max_len is None else max_len
    mask = torch.arange(max_l).unsqueeze(0).repeat(batch_size,1).to(lens.device)>lens.unsqueeze(1)
    return mask.unsqueeze(-1)

def read_phn_attr(phn_attr_pth, neg_val=0):
    df = pd.read_csv(phn_attr_pth, index_col=0, sep=SEP)
    attr = df.to_numpy()
    attr[attr==0] = neg_val
    attr = np.concatenate([np.zeros((PRESERVE_INDICES, attr.shape[1])), attr])
    return attr

def get_audio_duration(path):
    y, sr = sf.read(path)
    return len(y) / sr


