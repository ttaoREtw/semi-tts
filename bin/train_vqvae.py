import os
import json
import math
import torch
import numpy as np
from functools import partial

from src.vqvae import VQVAE, FRAME_PHN_RATIO
from src.optim import Optimizer
from src.solver import BaseSolver
from src.util import human_format, freq_loss, feat_to_fig, cal_per, cal_ppx, data_to_bar
from src.data import load_dataset

LISTEN_N_EXAMPLES = 6 # How many examples to show in tensorboard
SPEC_PAD_VALUE = 0 # Spectrogram was in log-scale
ATTENTION_PLOT_STEP = 500
CKPT_STEP = 10000
EPS = 1e-10

class VqvaeTrainer(BaseSolver):
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        # Init settings
        self.step = 0
        self.best_tts_loss = 100.0
        self.best_per = 2.0
        self.asr_weight = self.config['hparas']['asr_weight']
        self.tts_weight = self.config['hparas']['tts_weight']
        self.unpair_text_start_step = config['hparas']['unpair_text_start_step']
        self.unpair_text_weight = self.config['hparas']['unpair_text_weight']
        self.unpair_speech_start_step = config['hparas']['unpair_speech_start_step']
        self.unpair_speech_weight = self.config['hparas']['unpair_speech_weight']

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
    
        #return mel.to(self.device, non_blocking=True),\
        #       aug_mel.to(self.device, non_blocking=True),\
        #       linear.to(self.device, non_blocking=True),\
        #       text.to(self.device, non_blocking=True),\
        #       sid.to(self.device, non_blocking=True)
    
    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.verbose(['Loading data... large corpus may took a while.'])
        self.unpair_set, self.pair_set, self.dev_set, self.test_set, self.audio_converter, self.tokenizer, data_msg = \
                load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, **self.config['data'])
        self.pair_iter = iter(self.pair_set)
        self.unpair_iter = iter(self.unpair_set)
        self.dev_iter = iter(self.dev_set)
        # Feature statics
        self.n_mels, self.linear_dim = self.audio_converter.feat_dim
        self.vocab_size = self.tokenizer.vocab_size
        self.n_spkr = len(json.load(open(self.config['data']['corpus']['spkr_map'])))
        self.verbose(data_msg)

    def set_model(self):
        ''' Setup Audio AE-model and optimizer '''
        # Model
        self.model = VQVAE(self.n_mels, self.linear_dim, self.vocab_size, self.n_spkr, **self.config['model']).to(self.device)
        self.n_frames_per_step = self.model.n_frames_per_step
        self.verbose(self.model.create_msg())

        # Objective
        self.freq_loss = partial(
            freq_loss, 
            sample_rate=self.audio_converter.sr, 
            n_mels=self.audio_converter.n_mels,
            loss=self.config['hparas']['freq_loss_type'],
            differential_loss=self.config['hparas']['differential_loss'],
            emphasize_linear_low=self.config['hparas']['emphasize_linear_low']
            )
        self.ctc_loss = torch.nn.CTCLoss()
        self.stop_loss = torch.nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = Optimizer(self.model.parameters(),**self.config['hparas'])
        self.verbose(self.optimizer.create_msg())
        ### ToDo : unsup first?
        self.verbose('           | ASR weight = {}\t| start step = {}'.format(self.asr_weight,0))
        self.verbose('           | TTS weight = {}\t| start step = {}'.format(self.tts_weight,0))
        self.verbose('           | Txt weight = {}\t| start step = {}'.format(self.unpair_text_weight,
                                                                               self.unpair_text_start_step))
        self.verbose('           | Sph weight = {}\t| start step = {}'.format(self.unpair_speech_weight,
                                                                               self.unpair_speech_start_step))
        # ToDo: load pre-trained model
        if self.paras.load:
            ckpt = torch.load(self.paras.load, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_opt_state_dict(ckpt['optimizer'])
            self.step = ckpt['global_step']
            self.verbose('Load ckpt from {}, restarting at step {}'.format(self.paras.load,self.step))
    
    def exec(self):
        self.verbose(['Total training steps {}.'.format(human_format(self.max_step))])
        self.timer.set()
        unpair_speech_loss, unpair_text_loss, unsup_pred, unsup_trans, unsup_align = None, None, None, None, None
        ctc_nan_flag, ignore_speech_flag = 0,0
        tok_usage,gt_usage = [],[]
        cnter = {
            'ctc_nan':0,
            'unp_sph':0,
            'unp_txt':0
        }
        

        while self.step < self.max_step:
            # --------------------- Load data ----------------------- #
            # Unpair setting
            unpair_mel, unpair_aug_mel, unpair_linear, unpair_text, unpair_sid = None,None,None,None,None
            post_pred, asr_post_loss = None,None # For ASR postnet only
            use_unpair_text = self.unpair_text_weight>0 and self.step>self.unpair_text_start_step
            use_unpair_speech = self.unpair_speech_weight>0 and self.step>self.unpair_speech_start_step

            tf_rate = self.optimizer.pre_step(self.step)     # Catch the returned tf_rate if needed
            # ToDo : change # of sup. step = 2 x # of unsup. step ?
            mel, aug_mel, linear, text, sid = self.fetch_data(iter_name='pair_iter')

            # Load unpaired data only when use_unpair_xxx == True
            if self.step % 2 == 0: #2
            # if True:
                # ASR first
                speech_first = True
                if use_unpair_speech:                
                    unpair_mel, unpair_aug_mel, unpair_linear, unpair_text, unpair_sid = \
                                                    self.fetch_data(iter_name='unpair_iter')
            else:
                # TTS first
                speech_first = False
                if use_unpair_text:
                    cnter['unp_txt'] += 1
                    unpair_mel, unpair_aug_mel, unpair_linear, unpair_text, unpair_sid = \
                                                    self.fetch_data(iter_name='unpair_iter')

            total_loss = 0
            bs = len(mel)
            self.timer.cnt('rd')
            try:
                # ----------------------- Forward ------------------------ #
                if speech_first:
                    # Cycle : speech -> text -> speech
                    pair_prob, _, unpair_prob, unpair_latent, unpair_latent_len, pair_post_prob, _ = \
                                self.model.speech_to_text(paired_mel=aug_mel, unpaired_mel= unpair_aug_mel)

                    # Check to involve unsupervised Speech2Speech
                    if unpair_latent is not None:
                        # ASR output is the representataion for speech2speech
                        cnter['unp_sph'] += 1
                        ignore_speech_cycle = False
                        unpaired_teacher = unpair_mel
                    else:
                        # ASR output is all blank (cannot be passed to TTS) only paired text is used
                        ignore_speech_cycle = True
                        unpaired_teacher = None

                    # text -> speech
                    pair_mel_pred, pair_linear_pred, pair_align, _, \
                    unpair_mel_pred, unpair_linear_pred, unpair_align, _ =\
                                self.model.text_to_speech(paired_text = text, 
                                                          paired_sid=sid,
                                                          unpaired_sid=unpair_sid,
                                                          unpaired_latent = unpair_latent,
                                                          unpaired_text= None,
                                                          unpaired_latent_len = unpair_latent_len,
                                                          paired_teacher = mel,
                                                          unpaired_teacher = unpaired_teacher,
                                                          tf_rate = tf_rate
                                                         )
                else:
                    # Cycle : text -> speech -> text
                    pair_mel_pred, pair_linear_pred, pair_align, _, \
                    unpair_mel_pred, unpair_linear_pred, unpair_align, _ =\
                                self.model.text_to_speech(paired_text=text, 
                                                          paired_sid=sid,
                                                          unpaired_sid=unpair_sid,
                                                          unpaired_latent=None,
                                                          unpaired_text=unpair_text,
                                                          unpaired_latent_len=None,
                                                          paired_teacher=mel,
                                                          unpaired_teacher=None,
                                                          tf_rate=tf_rate
                                                         )
                    if use_unpair_text:
                        unpair_mel_pred = unpair_mel_pred.detach() # Stop-grad for tts in text2text
                    pair_prob, _, unpair_prob, unpair_latent, unpair_latent_len, pair_post_prob, _ = \
                                self.model.speech_to_text(paired_mel=aug_mel, 
                                                          unpaired_mel=unpair_mel_pred,  #None, #unpair_mel_pred, #None, #unpaired_mel= unpair_mel_pred,
                                                          using_fake_mel=use_unpair_text)

                # Paired ASR loss
                asr_loss = self.compute_ctcloss(aug_mel,pair_prob,text)
                if self.model.use_asr_postnet:
                    total_loss = total_loss + self.asr_weight*(1-self.model.asr_postnet_weight)*asr_loss
                    asr_post_loss = self.compute_ctcloss(aug_mel,pair_post_prob,text,apply_log=False)
                    total_loss = total_loss + self.asr_weight*self.model.asr_postnet_weight*asr_post_loss
                else:
                    total_loss = total_loss + self.asr_weight*asr_loss
                if math.isnan(asr_loss) or math.isinf(asr_loss):
                    cnter['ctc_nan'] += 1
                    asr_loss = 0

                # Paired TTS loss
                mel_loss = self.freq_loss(pair_mel_pred, mel)
                linear_loss = self.freq_loss(pair_linear_pred, linear)
                tts_loss = mel_loss + linear_loss
                total_loss = total_loss + self.tts_weight*tts_loss

                # Unpaired loss
                if speech_first:
                    # Unpaired speech reconstruction loss
                    if not ignore_speech_cycle:
                        unpair_speech_loss = self.freq_loss(unpair_mel_pred, unpair_mel) +\
                                            self.freq_loss(unpair_linear_pred, unpair_linear)
                        #total_loss += self.unpair_speech_weight*unpair_speech_loss
                        if self.step > self.unpair_speech_start_step:
                            total_loss += self.unpair_speech_weight*unpair_speech_loss
                elif use_unpair_text:
                    # Unpaired text reconstruction loss
                    ctc_input = (unpair_prob+EPS).transpose(0,1).log()
                    if self.paras.actual_len:
                        asr_input_len = (unpair_text!=0).sum(dim=-1)*FRAME_PHN_RATIO
                        asr_input_len = asr_input_len + asr_input_len%self.model.n_frames_per_step
                        ctc_len = 1+(asr_input_len//self.model.time_reduce_factor)
                    else:
                        ctc_len = torch.LongTensor([unpair_prob.shape[1]]*unpair_prob.shape[0]).to(device=self.device)
                    unpair_text_loss = self.ctc_loss( ctc_input, unpair_text.to_sparse().values(),
                                                      ctc_len, torch.sum(unpair_text!=0,dim=-1))
                    if math.isnan(unpair_text_loss) or math.isinf(unpair_text_loss):
                        cnter['ctc_nan'] += 1
                        unpair_text_loss = 0
                    total_loss += self.unpair_text_weight*unpair_text_loss

                # VQ-loss
                # if vq_loss>0:
                #     total_loss += self.model.vq_weight*vq_loss
                # if commit_loss>0:
                #     total_loss += self.model.commit_weight*commit_loss

                # Statics (over unsup. speech only)
                if speech_first and use_unpair_speech:
                    unsup_pred = unpair_prob.argmax(dim=-1).cpu()
                    unsup_trans = unpair_text.cpu()
                    tok_usage += unsup_pred.flatten().tolist()
                    gt_usage += unsup_trans.flatten().tolist()
                    if unpair_align is not None:
                        unsup_align = unpair_align.detach().cpu()
                    else:
                        unsup_align = [None]*bs

                self.timer.cnt('fw')

                # ----------------------- Backward ------------------------ #
                grad_norm = self.backward(total_loss)
                # For debugging
                # if math.isnan(grad_norm):
                  # import IPython
                  # IPython.embed()
                self.step+=1

                # Log
                if (self.step==1) or (self.step%self._PROGRESS_STEP==0):
                    self.progress('Tr stat | Loss - {:.2f} (CTC-nan/unp-sph/unp-txt={}/{}/{}) | Grad. Norm - {:.2f} | {} '\
                                  .format(total_loss.cpu().item(), cnter['ctc_nan'], cnter['unp_sph'], cnter['unp_txt'],
                                          grad_norm, self.timer.show()))
                    self.write_log('txt_loss',{'pair':asr_loss.item() if asr_loss is not None else None,
                                               'unpair':unpair_text_loss.item() if unpair_text_loss is not None else None,
                                               'post':asr_post_loss.item() if asr_post_loss is not None else None })
                    self.write_log('speech_loss',{'pair':tts_loss.item() if tts_loss is not None else None,
                                                'unpair':unpair_speech_loss.item() if unpair_speech_loss is not None else None})
                    #self.write_log('stop_err',{'tr':stop_err})
                    # if commit_loss>0:
                    #     self.write_log('commit',{'tr':commit_loss})
                    # if vq_loss>0:
                    #     self.write_log('commit',{'vq':vq_loss})
                    # self.write_log('temperature',{'temp':self.model.codebook.temp.data})
                    # self.write_log('ppx',{'tr':cal_ppx(p_code)})
                    for k in cnter.keys():
                        cnter[k] = 0
                    if (self.step==1) or (self.step%ATTENTION_PLOT_STEP ==0):
                        align = pair_align.cpu() # align shape BxDsxEs
                        sup_pred = pair_prob.argmax(dim=-1).cpu()
                        sup_trans = text.cpu()
                        if self.model.use_asr_postnet:
                            post_pred = pair_post_prob.argmax(dim=-1).cpu()
                        self.write_log('per',{'pair': cal_per(sup_pred,sup_trans),
                                              'unpair':cal_per(unsup_pred,unsup_trans),
                                              'post':cal_per(post_pred,sup_trans)})
                        self.write_log('unpair_hist',data_to_bar(tok_usage,gt_usage,self.vocab_size,self.tokenizer._vocab_list))
                        for i in range(LISTEN_N_EXAMPLES):
                            self.write_log('pair_align{}'.format(i),feat_to_fig(align[i].cpu().detach()))
                            if unsup_align is not None and unsup_align[i] is not None:
                                self.write_log('unpair_align{}'.format(i),feat_to_fig(unsup_align[i].cpu().detach()))
                        tok_usage,gt_usage = [],[]

                # Validation
                if (self.step==1) or (self.step%self.valid_step == 0):
                    self.validate()

                # End of step
                self.timer.set()
                if self.step > self.max_step:break

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self.verbose('WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                else:
                    print(repr(e))
                    errorout()
                

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

    def compute_ctcloss(self, model_input, model_output, target, apply_log=True):
        if apply_log:
            ctc_input = (model_output+EPS).transpose(0,1).log()
        else:
            ctc_input = model_output.transpose(0,1)

        if self.paras.actual_len:
            asr_input_len = torch.sum((model_input==SPEC_PAD_VALUE).long().sum(dim=-1)!=model_input.shape[-1],dim=-1)
            ctc_len = asr_input_len//self.model.time_reduce_factor
            ctc_target = target
        else:
            ctc_target = target.to_sparse().values()
            ctc_len = torch.LongTensor([model_output.shape[1]]*model_output.shape[0]).to(device=self.device)
        return self.ctc_loss( ctc_input, ctc_target,
                              ctc_len, torch.sum(target!=0,dim=-1))

