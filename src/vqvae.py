import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from src.asr import CTC as ASR
from src.asr import ASRPostnet
from src.tts import Tacotron2 as TTS
from src.embed import L2Embedding, SeperateEmbedding
from src.module import SpeakerEncoder

from src.util import get_sinusoid_encoding_table, get_audio_feat_mask, get_seq_mask

PRETRAINED_ENCODER_PREFIX = 'encoder.'
PRETRAINED_DECODER_PREFIX = 'decoder.'        # TTAO: Try load pretrained para. w/ this
PRETRAINED_POSTNET_PREFIX = 'postnet.'
FRAME_BLANK_TXT_RATIO = 4                     # TXT is actually twice length due to zero insertion
                                              # should be at least 4
FRAME_PHN_RATIO = 6.0
SPEC_PAD_VALUE = 0 # Spectrogram was in log-scale

class VQVAE(nn.Module):
    """VQVAE
    Procedure:
        mfcc segment -> SegmentAggregator + ASR -> fake phoneme vectors -> TTS -> mel-spectrogram
    """
    def __init__(self, n_mels, linear_dim, vocab_size, n_spkr, encoder, codebook, decoder, spkr_latent_dim, max_frames_per_phn, 
        stop_threshold, asr_postnet_weight=0.0, txt_update_codebook=False, pretrained_asr=None, pretrained_emb=None, 
        pretrained_tts=None):
        super().__init__()
        # Setup attributes
        self.in_dim = n_mels
        self.vocab_size = vocab_size
        self.n_spkr = n_spkr
        self.n_mels = n_mels
        self.linear_dim = linear_dim
        self.spkr_latent_dim = spkr_latent_dim
        self.stop_threshold = stop_threshold
        self.max_frames_per_phn = max_frames_per_phn
        self.txt_update_codebook = txt_update_codebook
        
        self.code_bone = codebook.pop('bone')
        self.latent_dim = codebook['latent_dim']
        self.commit_weight = codebook['commit_weight']
        self.vq_weight = codebook['vq_weight']
        self.n_frames_per_step = decoder['decoder']['n_frames_per_step']
        
        # ----------------- ASR model -----------------
        self.asr = ASR(n_mels, self.latent_dim, **encoder)
        self.time_reduce_factor = self.asr.time_reduce_factor
        self.use_asr_postnet = asr_postnet_weight>0
        if self.use_asr_postnet:
            self.asr_postnet_weight = asr_postnet_weight
            self.asr_postnet = ASRPostnet(self.latent_dim,self.latent_dim)

        # ----------------- Latent code ---------------
        if self.code_bone == 'l2':
            self.codebook = L2Embedding(vocab_size, False, **codebook)
        elif self.code_bone == 'seperate':
            self.codebook = SeperateEmbedding(vocab_size, False, **codebook)
        else:
            raise NotImplementedError

        # ------------- speaker embedding -------------
        self.spkr_embed = nn.Embedding(self.n_spkr, spkr_latent_dim)
        # self.spkr_enc = SpeakerEncoder(n_mels, spkr_latent_dim, **spkr_encoder)
        
        # ----------------- TTS model -----------------
        self.tts = TTS( n_mels, self.linear_dim, self.codebook.out_dim, self.spkr_latent_dim, decoder)

        # Load init. weights
        self.pretrain_asr = pretrained_asr is not None and pretrained_asr != ''
        if self.pretrain_asr:
            old_asr = torch.load(pretrained_asr)['model']
            old_asr = OrderedDict([(i[0].replace(PRETRAINED_ENCODER_PREFIX,''),i[1]) for i in old_asr.items()]) # rename parameters to match vqvae
            missing, _ = self.asr.load_state_dict(old_asr, strict=False)
            assert missing==[], 'Missing pretrained para. {}'.format(missing)
        self.pretrained_emb = pretrained_emb is not None and pretrained_emb != ''
        if self.pretrained_emb:
            old_emb = torch.load(pretrained_asr)['model']
            self.codebook.load_pretrained_embedding(old_emb)
        self.pretrained_tts = pretrained_tts is not None and pretrained_tts != ''
        if self.pretrained_tts:
            old_tts = torch.load(pretrained_tts)['model']
            old_tts = OrderedDict([(i[0].replace(PRETRAINED_DECODER_PREFIX,''),i[1]) for i in old_tts.items()])
            missing, _ = self.tts.decoder.load_state_dict(old_tts, strict=False)   # TTAO: are ALL weights included in pre-training?
            assert missing==[], 'Missing pretrained para. {}'.format(missing)
            old_postnet = OrderedDict([(i[0].replace(PRETRAINED_POSTNET_PREFIX,''),i[1])\
                                          for i in old_tts.items() if PRETRAINED_POSTNET_PREFIX in i[0]])
            missing, _ = self.tts.postnet.load_state_dict(old_postnet, strict=False)   # TTAO: are ALL weights included in pre-training?
            assert missing==[], 'Missing pretrained para. {}'.format(missing)

    def create_msg(self):
        msg = []
        msg.append('Model spec.| Codebook size = {}\t| Codebook dim = {}\t| VQ-weight = {}'\
                   .format(self.vocab_size, self.latent_dim, self.vq_weight))
        msg.append(self.codebook.create_msg())
        msg.append('           | Commitment-weight = {}'\
                   .format(self.commit_weight))
        msg.append('           | Enc reduce = {}\t| Dec n frames/sep post = {}/{}\t| Pretrained Enc/Emb/Dec = {}/{}/{}'\
                   .format(self.time_reduce_factor, self.n_frames_per_step, self.tts.separate_postnet, 
                           self.pretrain_asr, self.pretrained_emb, self.pretrained_tts))
        if self.use_asr_postnet:
            msg.append('           | ASR PostNet enabled, weight = {}\t'.format(self.asr_postnet_weight))
        return msg

    def speech_to_text(self, paired_mel, unpaired_mel, using_fake_mel=False):
        # concat batch if there's unpaired
        use_unpaired = unpaired_mel is not None
        if use_unpaired:
            paired_mel_bs, all_mel = self.padded_concat(paired_mel,unpaired_mel)
        else:
            all_mel = paired_mel
            paired_mel_bs = len(paired_mel)

        # Forward through ASR
        enc_latent = self.asr(all_mel)
        paired_post_prob = self.asr_postnet(enc_latent[:paired_mel_bs]) if self.use_asr_postnet else None
        first_n_real_mel = len(paired_mel) if using_fake_mel else 0
        p_code, quantized_latent, _, _ = self.codebook(enc_latent, first_n_real_mel)

        # Unpack
        if use_unpaired:
            pair_prob = p_code[:paired_mel_bs]
            pair_latent = quantized_latent[:paired_mel_bs]
            unpair_prob = p_code[paired_mel_bs:]
            unpair_latent = quantized_latent[paired_mel_bs:]
            # Trim repeated and blank (for unpaired only)
            trim_out = self.mean_forward(unpair_prob, unpair_latent)
            # Ignore unpaired speech if all-blank sample exists
            if trim_out is not None:
                unpair_latent, unpair_latent_len = trim_out
            else:
                unpair_latent, unpair_latent_len = None, None            
        else:
            pair_prob = p_code
            pair_latent = quantized_latent
            unpair_prob = None
            unpair_latent = None
            unpair_latent_len = None

        return  pair_prob, pair_latent, unpair_prob, unpair_latent, unpair_latent_len, paired_post_prob, _

    def text_to_speech(self, paired_text, paired_sid, unpaired_sid, unpaired_latent, unpaired_text, unpaired_latent_len,
                       paired_teacher, unpaired_teacher, tf_rate):
        # ToDo : see if unpaired_latent_len is required
        # Get phn embedding
        paired_latent = self.codebook.inference(paired_text)
        
        # concat batch if there's unpaired
        if unpaired_text is not None:
            # Text2text cycle
            assert unpaired_latent is None
            use_unpaired = True
            unpaired_latent = self.codebook.inference(unpaired_text)
            paired_latent_bs, all_latent = self.padded_concat(paired_latent,unpaired_latent)
            paired_ts = paired_teacher.shape[1]
            unpaired_ts = int(FRAME_PHN_RATIO*unpaired_text.shape[1])
            unpaired_ts += unpaired_ts%self.n_frames_per_step # make sure max_step%r==0
            unpair_max_frame = unpaired_ts
            # Note : no teacher so mel might be too short for CTC
            all_teacher = paired_teacher # No teacher for unpaired, |teacher|<|all_latent|
            #spkr_enc = self.spkr_enc(paired_ref_audio)
            all_spkr_latent = torch.cat([self.spkr_embed(paired_sid), self.spkr_embed(unpaired_sid)], dim=0)
        elif unpaired_latent is not None:
            # Speech2speech cycle
            use_unpaired = True
            paired_latent_bs, all_latent = self.padded_concat(paired_latent,unpaired_latent)
            paired_ts = paired_teacher.shape[1]
            unpaired_ts = unpaired_teacher.shape[1]
            unpair_max_frame = None # teacher gives the exact decode step
            paired_teacher_bs, all_teacher = self.padded_concat(paired_teacher,unpaired_teacher)
            #all_spkr_latent = torch.cat([self.spkr_enc(paired_ref_audio), self.spkr_enc(unpaired_teacher)], dim=0)
            all_spkr_latent = torch.cat([self.spkr_embed(paired_sid), self.spkr_embed(unpaired_sid)], dim=0)
        else:
            use_unpaired = False
            all_latent = paired_latent
            all_teacher = paired_teacher
            #all_spkr_latent = self.spkr_enc(paired_ref_audio)
            all_spkr_latent = self.spkr_embed(paired_sid)
            unpair_max_frame = None # No unpaired text in this case

        # Forward (Note: forwarding unpaired together with padding may affect BN)
        mel, linear, align, stop = self.tts( all_latent, None, all_teacher, all_spkr_latent,
                                             tf_rate=tf_rate, unpair_max_frame=unpair_max_frame)

        # Unpack
        if use_unpaired:
            pair_mel = mel[:paired_latent_bs,:paired_ts]
            pair_linear = linear[:paired_latent_bs,:paired_ts]
            pair_align = align[:paired_latent_bs,:paired_ts]
            pair_stop = stop[:paired_latent_bs]
            unpair_mel = mel[paired_latent_bs:,:unpaired_ts]
            unpair_linear = linear[paired_latent_bs:,:unpaired_ts]
            unpair_align = align[paired_latent_bs:,:unpaired_ts]
            unpair_stop = stop[paired_latent_bs:]
        else:
            pair_mel = mel
            pair_linear = linear
            pair_align = align
            pair_stop = stop
            unpair_mel = None
            unpair_linear = None
            unpair_align = None
            unpair_stop = None

        return pair_mel, pair_linear, pair_align, pair_stop, \
               unpair_mel, unpair_linear, unpair_align, unpair_stop

    def tts_grad_switch(self, state, mode='encoder'):
        state = state.lower()
        assert state in ['on', 'off']
        mode = mode.lower()
        assert mode in ['encoder', 'all']
        paras = self.tts.parameters() if mode == 'all' else self.tts.encoder.parameters()
        for p in paras:
            p.required_grad = True if state == 'on' else False

    def mean_forward(self, p_code, latent):
        B,T,D = latent.shape

        batch_latent = []
        trimmed_len = []
        latent_idx_seq = p_code.argmax(dim=-1)
        for b,idx_seq in enumerate(latent_idx_seq):
            idx_seq = idx_seq.cpu().tolist()
            last_idx = idx_seq[0]
            last_pos = 0
            cur_seq = []
            # Scan through seq, filter out blank and merge all duplicated token
            for t,idx in enumerate(idx_seq):
                if last_idx!=idx or ((t-last_pos)>self.max_frames_per_phn):
                # if last_idx!=idx:
                    if last_idx!=0:
                        cur_seq.append(latent[b,last_pos:t,:].mean(dim=0))
                    last_idx = idx
                    last_pos = t

            # Append last token if non-blank
            if last_idx!=0:
                if last_pos != (T-1):
                    # Last token is end of consecutive non-blank tokens
                    cur_seq.append(latent[b,last_pos:,:].mean(dim=0))
                else:
                    # Last token is non-blank and unique                        
                    cur_seq.append(latent[b,t,:])
                    
            # return None if there's a all-blank sample
            if len(cur_seq) == 0:
                return None
            # Stack as trimmed seq
            trimmed_len.append(len(cur_seq))
            batch_latent.append(torch.stack(cur_seq,dim=0))

        batch_latent = nn.utils.rnn.pad_sequence(batch_latent, batch_first=True)
        trimmed_len = torch.LongTensor(trimmed_len).to(batch_latent.device)

        return batch_latent, trimmed_len

    def padded_concat(self, pair, unpair):
        pair_bs = pair.shape[0]
        pair_ts = pair.shape[1]
        unpair_ts = unpair.shape[1]
        if pair_ts>unpair_ts:
            unpair = torch.cat([unpair,
                                torch.zeros_like(pair)[:,:(pair_ts-unpair_ts)]],dim=1)
        elif pair_ts<unpair_ts:
            pair = torch.cat([pair,
                              torch.zeros_like(unpair)[:,:(unpair_ts-pair_ts)]],dim=1)
        
        concat_batch = torch.cat([pair,unpair],dim=0)
        return pair_bs, concat_batch
