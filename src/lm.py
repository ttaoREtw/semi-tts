#import kenlm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


#from src.module import PreNet,Decoder,CBHG
from src.embed import VQEmbedding, DualEmbedding
from src.module import Decoder, CBHG, TransformerEncoder, SpeechEncoder, ASRDecoder
from src.util import get_audio_feat_mask, get_seq_mask

EPS = 1e-10 # to avoid log nan
INF= 1e10

class AudioLM(nn.Module):
    ''' Audio LM for pretraining, will serve as init weight for TTS decoder later'''
    def __init__(self, feat_dim, n_frames_per_step, enc_embed_dim, prenet_dim, 
                 query_rnn_dim, dec_rnn_dim, p_query_dropout, p_dec_dropout, 
                 attn_dim, n_location_filters, location_kernel_size, prenet_dropout,
                 loc_aware, use_summed_weights):
        super().__init__()
        # Setup feature dim.
        self.out_dim = feat_dim[0]
        self.output_linear = feat_dim[1] is not None
        self.n_frames_per_step = n_frames_per_step
        self.enc_embed_dim = enc_embed_dim
        self.prenet_dim = prenet_dim
        self.query_rnn_dim = query_rnn_dim
        self.dec_rnn_dim = dec_rnn_dim
        self.p_query_dropout = p_query_dropout
        self.p_dec_dropout = p_dec_dropout
        self.attn_dim = attn_dim
        self.n_location_filters = n_location_filters
        self.location_kernel_size = location_kernel_size
        self.loc_aware = loc_aware
        self.use_summed_weights = use_summed_weights

        # Decoder  (to be used in TTS)
        self.decoder = Decoder(
            n_mels=self.out_dim,
            n_frames_per_step=n_frames_per_step,
            enc_embed_dim=enc_embed_dim,
            prenet_dim=prenet_dim,
            prenet_dropout=prenet_dropout,
            query_rnn_dim=query_rnn_dim,
            dec_rnn_dim=dec_rnn_dim,
            query_dropout=p_query_dropout,
            dec_dropout=p_dec_dropout,
            attn_dim=attn_dim,
            n_location_filters=n_location_filters,
            location_kernel_size=location_kernel_size,
            loc_aware=loc_aware,
            use_summed_weights=use_summed_weights,
            pretrain=True)
        
        if self.output_linear:
            self.linear_dim = feat_dim[1]
            self.postnet = nn.Sequential(
                CBHG(self.out_dim, K=8),
                # CBHG output size is 2 * input size
                nn.Linear(self.out_dim * 2, self.linear_dim))
    
    def create_msg(self):
        # Messages for user
        ### ttao : format strings that shows info you think is important
        msg = ['Model spec.| AudioLM [query/dec]_rnn_dim = {}/{}, frames/step = {}'.format(
            self.query_rnn_dim, self.dec_rnn_dim, self.n_frames_per_step)]
        return msg
    
    def forward(self, x):
        # Input shape (BxTxself.out_dim)
        B, T, D = x.shape
        linear = None
        
        empty_memory = torch.zeros(B,2,self.enc_embed_dim).to(x.device) # Pretraining does not require memory
        empty_len = torch.LongTensor([2]*B).to(x.device)
        mel, _ = self.decoder(empty_memory, empty_len, x, tf_rate=1.0)
        if self.output_linear:
            # detach() or not (?)
            linear = self.postnet(mel)

        # mask output
        '''
        actual_lengths = torch.sum(torch.sum(x, dim=-1) != 0, dim=-1)
        mask_mel = get_audio_feat_mask(actual_lengths, self.n_frames_per_step, self.out_dim)
        mel.data.masked_fill_(mask_mel, 0)
        if self.output_linear:
            mask_linear = get_audio_feat_mask(actual_lengths, self.n_frames_per_step, self.linear_dim)
            linear.data.masked_fill_(mask_linear, 0)
        '''
        
        return mel, linear


class TextLM(nn.Module):
    ''' Text LM for ASR decoder '''
    def __init__(self, vocab_size, in_dim, encoder, codebook):
        super().__init__()
        self.vocab_size = vocab_size
        self.out_dim = codebook['latent_dim']
        self.normalize = codebook['emb_norm']
        self.code_bone = codebook.pop('bone')

        if self.code_bone == 'ema':
            self.emb = VQEmbedding(vocab_size, True, **codebook)
        elif self.code_bone == 'vq':
            self.emb = VQEmbedding(vocab_size, False, **codebook)
        elif self.code_bone == 'vanilla':
            self.emb = nn.Embedding(vocab_size, self.out_dim)
        elif self.code_bone == 'dual':
            # By default, dual emb. uses ema to update codebook
            self.emb = DualEmbedding(vocab_size, True, **codebook)
        else:
            raise NotImplementedError
        
        self.encoder = ASRDecoder(in_dim, vocab_size, self.out_dim, **encoder)
    
    def return_temp(self):
        if hasattr(self.emb, 'temp'):
            return self.emb.temp.data
        return None
    
    def create_msg(self):
        # Messages for user
        msg = ['Model spec.| TextLM Emb. dim = {} , bone = {}'.format(self.out_dim, self.code_bone)]
        return msg

    def forward(self, txt, txt_lens):
        enc_logits = self.encoder.pretrain( txt, txt_lens, self.normalize)
        if self.code_bone == 'vanilla':
            emb_matrix = F.normalize(self.emb.weight, dim=-1) if self.normalize else self.emb.weight
            output_prob = torch.log_softmax(F.linear(enc_logits,emb_matrix),dim=-1)
        else:
            output_prob, _, _, _ = self.emb(enc_logits,get_seq_mask(txt_lens,enc_logits.size(1)))
            output_prob = (output_prob+EPS).log()

        return output_prob


class DenoisingLM(nn.Module):
    ''' Transformer-based Denoising Language Model '''
    def __init__(self, vocab_size, in_dim, encoder, codebook):
        super().__init__()
        self.in_dim = in_dim

        self.text_to_fake_wave = nn.Embedding(vocab_size, in_dim, padding_idx=0)

        self.dim = codebook['latent_dim']
        self.vocab_size = vocab_size
        self.emb_norm = codebook['emb_norm']
        self.emb = nn.Embedding(vocab_size, self.dim)
        self.bone = encoder['bone'] 
        if self.bone == 'cnn':
            self.encoder = SpeechEncoder( in_dim, self.dim, **encoder)
        elif self.bone == 'Transformer':
            self.encoder = TransformerEncoder( in_dim, self.dim, **encoder)
        else:
            raise NotImplementedError

    def create_msg(self):
        # Messages for user
        msg = ['Model spec.| Encoder backbone = {}, output dim = {}'.format(self.bone ,self.dim)]
        return msg

    def forward(self, x, lens):
        # Foward through denoising lm
        if self.emb_norm:
            with torch.no_grad():
                self.emb.weight.div_(torch.norm(self.emb.weight, dim=-1, keepdim=True))

        phn_enc = self.text_to_fake_wave(x)
        enc_seq = self.encoder(phn_enc,lens)

        if self.emb_norm:
            enc_seq = F.normalize(enc_seq, p=2, dim=-1)

        # Compute output prob. based on negative L2 distance
        b,t,d = enc_seq.shape
        if self.emb_norm:
            outputs = F.log_softmax( F.linear(enc_seq,self.emb.weight), dim=-1)
        else:
            flat_seq = enc_seq.reshape(b*t,d)
            # L2 
            distance =   torch.sum(flat_seq.pow(2), dim=-1,keepdim=True) \
                       + torch.sum(self.emb.weight.pow(2), dim=-1)\
                       - 2 * torch.matmul(flat_seq, self.emb.weight.t())
                       
            outputs = F.log_softmax(-distance.view(b,t,-1),dim=-1)

        return outputs

class RNNLM(nn.Module):
    ''' RNN Language Model '''
    def __init__(self, vocab_size, emb_dim, module, n_head, dim, n_layers, dropout):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dropout = dropout>0
        if self.dropout:
            self.dp = nn.Dropout(dropout)
        self.rnn = getattr(nn, module.upper())(emb_dim, dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.post_rnn = nn.Linear(dim,emb_dim,bias=None)


    def create_msg(self):
        # Messages for user
        msg = ['Model spec.| RNNLM # of layers = {}, dim = {}'.format(self.n_layers,self.dim)]
        return msg

    def forward(self, x, lens, hidden=None):
        emb_x = self.emb(x)
        if self.dropout:
            emb_x = self.dp(emb_x)

        packed = nn.utils.rnn.pack_padded_sequence(emb_x, lens,batch_first=True)
        outputs, hidden = self.rnn(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = self.post_rnn(outputs)
        b,t,d = outputs.shape
        outputs = outputs.view(b*t,d)
        # L2 
        distance =   torch.sum(outputs.pow(2), dim=-1,keepdim=True) \
                   + torch.sum(self.emb.weight.pow(2), dim=-1)\
                   - 2 * torch.matmul(outputs, self.emb.weight.t())
                   
        outputs = F.log_softmax(distance.view(b,t,-1),dim=-1)

        return outputs

class NgramPrior(nn.Module):
    ''' N-gram LM as prior knowledge, backed by kemlm '''
    def __init__(self, vocab_size, start_step, path, n_gram, weight, reduction):
        super().__init__()
        
        # Construct transform for torch idx matrix -> N-gram prefix index
        self.n_gram = n_gram
        self.start_step = start_step
        self.path = path
        self.vocab_size = vocab_size
        self.weight = weight
        self.reduction = reduction
        v = self.vocab_size
        n = self.n_gram
        
        if self.n_gram >1 :
            # index to onehot [N-1,1] -> [N-1,V]
            self.ngram2idx = nn.Conv1d(1, 1, n-1, bias=False)
            for i in range(n-1):
                self.ngram2idx.weight[0,0,i] = v**(n-2-i)
            
            # Load n-gram prob. table
            ngram_distribution = torch.FloatTensor(np.load(path))+EPS
            self.ngram_distribution = nn.Embedding.from_pretrained(ngram_distribution, freeze=True)
        else:
            unigram_prob = torch.FloatTensor(np.load(path))+EPS
            self.register_buffer('ngram_distribution', torch.FloatTensor(unigram_prob))

    def create_msg(self):
        # Messages for user
        msg = ['Prior spec.| {}-Gram LM from {} with Lambda = {:.0e}'.format(self.n_gram,self.path,self.weight)]
        return msg

    def compute_loss(self, enc_prob, enc_len):
        ''' Input shape [BxTxV] '''
        # Compute prior distribution over vocab for every N-1 gram
        B,T,_ = enc_prob.shape

        if self.n_gram>1:
            with torch.no_grad():
                code_index = enc_prob.argmax(dim=-1)
                # Zero padding and start with <sos>
                code_index = torch.cat([torch.zeros(B,max(0,self.n_gram-2)).long().to(enc_prob.device),
                                        torch.ones(B,1).long().to(enc_prob.device),
                                        code_index[:,:-1]],dim=-1)
                ngram_idx = self.ngram2idx(code_index.float().unsqueeze(1)).squeeze(1).long()
                
                prior_prob = self.ngram_distribution(ngram_idx)
        else:
            prior_prob = self.ngram_distribution.unsqueeze(0).repeat(B,T,1)

        
        # Zero out padded part
        mask = get_seq_mask(enc_len,T)
        #enc_prob = enc_prob.masked_fill(mask,0.0) already masked
        #enc_prob = enc_prob
        #prior_prob = prior_prob.masked_fill(mask,0.0) # for Likelihood
        prior_prob = prior_prob.masked_fill(mask,EPS)
        lens = enc_len.float()
        '''
        # Compute Likeihood between prior & P_enc
        if self.reduction == 'token':
            # frame-wised KLD
            kld = - torch.sum(prior_prob*(enc_prob.log()),dim=-1)                  # Sum over Vocab
            kld = kld.sum(dim=-1)/lens                                             # Mean by length
            kld = kld.mean()                                                       # Mean by batch
        elif self.reduction == 'sentence':
            # Sentence-wised KLD
            enc_prob = enc_prob.sum(dim=1)/lens.unsqueeze(1)                       # Accu. prob. over frames
            prior_prob = prior_prob.sum(dim=1)/lens.unsqueeze(1)
            kld = -prior_prob*enc_prob.log()
            kld = kld.sum(dim=-1).mean()                                           # Sum over Vocab and Mean by batch
        elif self.reduction == 'batch':
            # Batch-wised KLD
            enc_prob = (enc_prob.sum(dim=1)/lens.unsqueeze(1)).mean(dim=0)         # Accu. prob. over whole batch
            prior_prob = (prior_prob.sum(dim=1)/lens.unsqueeze(1)).mean(dim=0)
            kld = -prior_prob*enc_prob.log()
            kld = kld.sum()                                                        # sum over Vocab
        else:
            raise NotImplementedError

        '''
        # Compute KLD (Note : ignore entropy of encoder)
        if self.reduction == 'token':
            # frame-wised KLD
            #kld = torch.sum(enc_prob*((enc_prob+EPS).log()-prior_prob.log()),dim=-1)     # Sum over Vocab
            kld = - torch.sum(enc_prob*prior_prob.log(),dim=-1)                    # Sum over Vocab
            kld = kld.sum(dim=-1)/lens                                             # Mean by length
            kld = kld.mean()                                                       # Mean by batch
        elif self.reduction == 'sentence':
            # Sentence-wised KLD
            enc_prob = enc_prob.sum(dim=1)/lens.unsqueeze(1)                       # Accu. prob. over frames
            prior_prob = prior_prob.sum(dim=1)/lens.unsqueeze(1)
            #kld = enc_prob*((enc_prob+EPS).log()-prior_prob.log())
            kld = -enc_prob*prior_prob.log()
            kld = kld.sum(dim=-1).mean()                                           # Sum over Vocab and Mean by batch
        elif self.reduction == 'batch':
            # Batch-wised KLD
            enc_prob = (enc_prob.sum(dim=1)/lens.unsqueeze(1)).mean(dim=0)         # Accu. prob. over whole batch
            prior_prob = (prior_prob.sum(dim=1)/lens.unsqueeze(1)).mean(dim=0)
            #kld = enc_prob*((enc_prob+EPS).log()-prior_prob.log())
            kld = -enc_prob*prior_prob.log()
            kld = kld.sum()                                                        # sum over Vocab
        else:
            raise NotImplementedError
        
        return kld
