import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.embed import L2Embedding as Embedding
from src.module import Encoder, Decoder, Postnet, CBHG, Linear
#from src.util import get_audio_feat_mask

class Tacotron2(nn.Module):
    """Tacotron2 text-to-speech model (w/o stop prediction)
    """
    def __init__(self, n_mels, linear_dim, in_embed_dim, spkr_embed_dim, paras):
        super(Tacotron2, self).__init__()
        self.n_mels = n_mels
        self.linear_dim = linear_dim
        if 'separate_postnet' in paras:
            self.separate_postnet = paras['separate_postnet']
        else:
            self.separate_postnet = False
        self.encoder = Encoder(in_embed_dim, **paras['encoder'])
        self.decoder = Decoder(
            n_mels, enc_embed_dim=self.encoder.enc_embed_dim, spkr_embed_dim=spkr_embed_dim, **paras['decoder'])
        self.prenet_dim = self.decoder.prenet_dim
        self.prenet_dropout = self.decoder.prenet_dropout
        self.loc_aware = self.decoder.loc_aware
        self.use_summed_weights = self.decoder.use_summed_weights
        self.n_frames_per_step = self.decoder.n_frames_per_step
        # Whether to use CBHG to convert mel to linear or not
        self.postnet = None
        if linear_dim is not None:
            self.postnet = nn.Sequential(
                CBHG(n_mels, K=8),
                # CBHG output size is 2 * input size
                nn.Linear(n_mels * 2, linear_dim))
        
    def forward(self, txt_embed, txt_lengths, teacher, spkr_embed, tf_rate=0.0, unpair_max_frame=None):
        """
        Arg:
            txt_embed: the output of TextEmbedding of shape (B, L, enc_embed_dim)
            txt_lengths: text lengths before padding (B)
            teacher: max_dec_step for inference. (B, T, n_mels) for training
            tf_rate: teacher forcing rate, `1.0` for pure teacher forcing.
        """
        enc_output = self.encoder(txt_embed, txt_lengths)
        mel_pred, alignment, stop = self.decoder(enc_output, txt_lengths, teacher, spkr_embed,
                                                 tf_rate=tf_rate, unpair_max_frame=unpair_max_frame)
        if self.separate_postnet:
            linear_pred = self.postnet(mel_pred.detach()) if self.postnet is not None else None # For demo
        else:
            linear_pred = self.postnet(mel_pred) if self.postnet is not None else None
        return mel_pred, linear_pred, alignment, stop

    def create_msg(self):
        msg = []
        msg.append('Model spec.| Model = `TACO-2`\t| Prenet dim = {}\t| Prenet dropout = {}\t'.format(
                    self.prenet_dim, self.prenet_dropout))
        msg.append('           | Loc. aware = {}\t| frames/step = {}\t| mel2linear = {}\t| sep_post = {}\t'.format(
                    self.loc_aware. self.n_frames_per_step, self.postnet is not None, self.separate_postnet))
        return msg

class Tacotron2withCodebook(nn.Module):
    def __init__(self, n_mels, linear_dim, vocab_size, paras_tts, paras_codebook):
        super(Tacotron2withCodebook, self).__init__()
        # Remember to pop 'bone'
        paras_codebook.pop('bone')
        self.codebook = Embedding(vocab_size, False, **paras_codebook)
        self.tts = Tacotron2(n_mels, linear_dim, self.codebook.out_dim, paras_tts)
        self.n_frames_per_step = self.tts.decoder.n_frames_per_step
        self.create_msg = self.tts.create_msg

    def forward(self, txt, txt_lengths, teacher, tf_rate=0.0):
        txt_embed = self.codebook.inference(txt)
        mel_pred, alignment, stop = self.tts(txt_embed, txt_lengths, teacher, tf_rate)
        return mel_pred, alignment, stop
 
