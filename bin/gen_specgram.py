import os
import json
import math
import torch
import numpy as np
import soundfile as sf
from os.path import join
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from src.vqvae import VQVAE, FRAME_PHN_RATIO
from src.optim import Optimizer
from src.solver import BaseSolver
from src.util import human_format, freq_loss, feat_to_fig, cal_per, cal_ppx, data_to_bar
from src.data import load_dataset

INFERENCE_MARGIN_FRAMES = 40 #27
LISTEN_N_EXAMPLES = 6 # How many examples to show in tensorboard
SPEC_PAD_VALUE = 0 # Spectrogram was in log-scale
ATTENTION_PLOT_STEP = 500
CKPT_STEP = 10000
EPS = 1e-10

class SpecgramGenerator(BaseSolver):
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)

    def fetch_data(self, iter_name):
        # Load from iterator
        mel = None
        while mel is None:
            try:
                mel, aug_mel, linear, sid, text = next(getattr(self,iter_name))
            except StopIteration:
                setattr(self,iter_name,iter(getattr(self,iter_name.replace('iter','set'))))

        # Pad to match n_frames_per_step (at least 1 frame padded)
        pad_len = self.n_frames_per_step - (mel.shape[1]%self.n_frames_per_step)
        mel = torch.cat([mel, SPEC_PAD_VALUE*torch.ones_like(mel)[:,:pad_len,:]], dim=1)
        linear = torch.cat([linear, SPEC_PAD_VALUE*torch.ones_like(linear)[:,:pad_len,:]], dim=1)

        return mel.to(self.device),\
               aug_mel.to(self.device),\
               linear.to(self.device),\
               text.to(self.device),\
               sid.to(self.device)
    
    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.unpair_set, self.pair_set, self.dev_set, self.test_set, self.audio_converter, self.tokenizer, data_msg = \
                load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, inference_stage=True, 
                             **self.config['data'])
        self.pair_iter = iter(self.pair_set)
        self.unpair_iter = iter(self.unpair_set)
        self.dev_iter = iter(self.dev_set)
        self.test_iter = iter(self.test_set)
        # Feature statics
        self.n_mels, self.linear_dim = self.audio_converter.feat_dim
        self.vocab_size = self.tokenizer.vocab_size
        self.n_spkr = len(json.load(open(self.config['data']['corpus']['spkr_map'])))

        self.filelist = {'pair': [], 'unpair': [], 'dev': [], 'test': []}
        self.filelist['pair'] = self.pair_set.dataset.table.index.tolist()
        self.filelist['unpair'] = self.unpair_set.dataset.table.index.tolist()
        self.filelist['dev'] = self.dev_set.dataset.table.index.tolist()
        self.filelist['test'] = self.test_set.dataset.table.index.tolist()
        # self.verbose(data_msg)

    def set_model(self):
        ''' Setup Audio AE-model and optimizer '''
        # Model
        self.model = VQVAE(self.n_mels, self.linear_dim, self.vocab_size, self.n_spkr, **self.config['model']).to(self.device)
        self.model.eval()
        self.n_frames_per_step = self.model.n_frames_per_step
        # self.verbose(self.model.create_msg())

        assert self.paras.load is not None
        # ToDo: load pre-trained model
        ckpt = torch.load(self.paras.load, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.step = ckpt['global_step']
        self.verbose('Load ckpt from {}, restarting at step {}'.format(self.paras.load,self.step))

    
    def exec(self):
        self.gen_specgram('test', self.logdir + '_%dk' % (self.step // 1000))
        

    def gen_specgram(self, split, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.verbose('Save outputs in {}'.format(output_dir))
        cnt = 0
        dataset = getattr(self, split + '_set')
        iter_name = split + '_iter'
        for i in range(len(dataset)):
            self.progress('Generating spectrogram for {} - {}/{}'.format(split, i+1, len(dataset)))
            mel, aug_mel, linear, text, sid = self.fetch_data(iter_name=iter_name)
            with torch.no_grad():
                pair_mel_pred, pair_linear_pred, pair_align, _, _, _, _, _ = \
                        self.model.text_to_speech(paired_text = text,
                                                  paired_sid=sid,
                                                  unpaired_sid=None,
                                                  unpaired_latent=None,
                                                  unpaired_text=None,
                                                  unpaired_latent_len=None,
                                                  paired_teacher=mel.shape[1] + INFERENCE_MARGIN_FRAMES,
                                                  unpaired_teacher=None,
                                                  tf_rate=0.0)
                mel_pred = pair_mel_pred
                linear_pred = pair_linear_pred
                align_pred = pair_align
                enc_step = (text!=0).sum(dim=-1).cpu().tolist()
                dec_step = [int(t_len*FRAME_PHN_RATIO)//self.n_frames_per_step for t_len in enc_step]
                if self.paras.gen_wav:
                    wavs, sr = self.audio_converter.feat_to_wave(linear_pred)

            for idx, (msp, sp, ali) in enumerate(zip(mel_pred, linear_pred, align_pred)):
                fname = self.filelist[split][cnt]
                np.save(join(output_dir, fname + '-mel.npy'), 
                        msp.cpu().numpy().astype(np.float32), allow_pickle=False)
                np.save(join(output_dir, fname + '-spec.npy'), 
                        sp.cpu().numpy().astype(np.float32), allow_pickle=False)                
                ali = ali[:dec_step[idx], :enc_step[idx]]
                np.save(join(output_dir, fname + '-align.npy'), ali.cpu().numpy())
                if self.paras.gen_wav:
                    sf.write(join(output_dir, fname + '-pred.wav'), wavs[idx], sr)
                cnt += 1

        self.verbose('Save {} spectorgram totally'.format(cnt))


    def validate(self):
        # Eval mode
        self.model.eval()
        dev_tts_loss, dev_per, dev_post_per, dev_stop_err = [], [], [], []

        for i in range(len(self.dev_set)):
            self.progress('Valid step - {}/{}'.format(i+1,len(self.dev_set)))
            # Fetch data
            mel, aug_mel, linear, text, sid = self.fetch_data(iter_name='dev_iter')

            # Forward model
            with torch.no_grad():
                # test ASR
                pair_prob, _, _, _, _, pair_post_prob, _ = self.model.speech_to_text(paired_mel=mel, unpaired_mel=None)
                dev_per.append(cal_per(pair_prob,text))
                if pair_post_prob is not None:
                    dev_post_per.append((cal_per(pair_post_prob,text)))

                # test TTS (Note: absolute dec step now)
                pair_mel_pred, pair_linear_pred, pair_align, _, _, _, _, _ = \
                        self.model.text_to_speech(paired_text = text,
                                                  paired_sid=sid,
                                                  unpaired_sid=None,
                                                  unpaired_latent=None,
                                                  unpaired_text=None,
                                                  unpaired_latent_len=None,
                                                  paired_teacher=mel.shape[1],
                                                  unpaired_teacher=None,
                                                  tf_rate=0.0)
                dev_tts_loss.append(self.freq_loss(pair_mel_pred, mel) + self.freq_loss(pair_linear_pred, linear))

            if i == len(self.dev_set)//2:
                # pick n longest samples in the median batch
                sample_txt = text.cpu()[:LISTEN_N_EXAMPLES]
                hyp = pair_prob.argmax(dim=-1).cpu()[:LISTEN_N_EXAMPLES]
                mel_p = pair_mel_pred.cpu()[:LISTEN_N_EXAMPLES]
                linear_p = pair_linear_pred.cpu()[:LISTEN_N_EXAMPLES]
                #post_mel_p = tts_pred.cpu()[:LISTEN_N_EXAMPLES,1] # PostNet product
                align_p = pair_align.cpu()[:LISTEN_N_EXAMPLES]
                sample_mel = mel.cpu()[:LISTEN_N_EXAMPLES]
                sample_linear = linear.cpu()[:LISTEN_N_EXAMPLES]
                
        # Ckpt if performance improves 
        dev_tts_loss = sum(dev_tts_loss)/len(dev_tts_loss)
        dev_per = sum(dev_per)/len(dev_per)
        dev_post_per = sum(dev_post_per)/len(dev_post_per) if len(dev_post_per)>0 else None
        #dev_stop_err = sum(dev_stop_err)/len(dev_stop_err)

        if self.paras.store_best_per:
            if dev_per < self.best_per:
                self.best_per = dev_per
                self.save_checkpoint('best_per.pth', dev_per)
            if (dev_post_per is not None) and (dev_post_per < self.best_per):
                self.best_per = dev_post_per
                self.save_checkpoint('best_post_per.pth', dev_post_per)
        else:
            if dev_tts_loss < self.best_tts_loss:
                self.best_tts_loss = dev_tts_loss
                if self.step>1:
                    self.save_checkpoint('tts_{}.pth'.format(self.step),dev_tts_loss)
            if dev_per < self.best_per:
                self.best_per = dev_per
                if self.step>1:
                    self.save_checkpoint('asr_{}.pth'.format(self.step),dev_per)
            if (dev_post_per is not None) and (dev_post_per < self.best_per):
                self.best_per = dev_post_per
                self.save_checkpoint('best_post_per.pth', dev_post_per) # Note: didnot recode best per from postnet or not

        if ((self.step>1) and (self.step%CKPT_STEP==0)) and not self.paras.store_best_per:
            # Regular ckpt
            self.save_checkpoint('step_{}.pth'.format(self.step),dev_tts_loss)

        # Logger 
        # Write model output (no G-F-lim if picking per)    
        for i,(m_p,l_p,a_p,h_p) in enumerate(zip(mel_p,linear_p,align_p, hyp)):
            self.write_log('hyp_text{}'.format(i), self.tokenizer.decode(h_p.tolist()))
            self.write_log('mel_spec{}'.format(i), feat_to_fig(m_p))
            self.write_log('linear_spec{}'.format(i), feat_to_fig(l_p))
            self.write_log('dv_align{}'.format(i), feat_to_fig(a_p))
            if not self.paras.store_best_per:
                self.write_log('mel_wave{}'.format(i), self.audio_converter.feat_to_wave(m_p))
                self.write_log('linear_wave{}'.format(i), self.audio_converter.feat_to_wave(l_p))
        # Write ground truth
        if self.step ==1:
            for i,(mel,linear,gt_txt) in enumerate(zip(sample_mel,sample_linear,sample_txt)):
                self.write_log('truth_text{}'.format(i), self.tokenizer.decode(gt_txt.tolist()))
                self.write_log('mel_spec{}_gt'.format(i), feat_to_fig(mel))
                self.write_log('mel_wave{}_gt'.format(i), self.audio_converter.feat_to_wave(mel))
                self.write_log('linear_spec{}_gt'.format(i), feat_to_fig(linear))
                self.write_log('linear_wave{}_gt'.format(i), self.audio_converter.feat_to_wave(linear))
        
        self.write_log('speech_loss',{'dev':dev_tts_loss})
        self.write_log('per',{'dev':dev_per, 'dev_post':dev_post_per})
        self.write_log('codebook',(self.model.codebook.embedding.weight.data,self.tokenizer._vocab_list))
        #self.write_log('stop_err',{'dev':dev_stop_err})
        # Resume training
        self.model.train()

