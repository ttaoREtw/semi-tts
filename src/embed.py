import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.util import read_phn_attr

EPS = 1e-5

class BaseEmbedding(nn.Module):
    """
    Base class of all embedding
    """
    def __init__(self, vocab_size, softmax, latent_dim, commit_weight, vq_weight, temp):
        super().__init__()
        # Required attributes
        self.vocab_size = vocab_size
        self.softmax = softmax
        self.latent_dim = latent_dim
        self.out_dim = latent_dim
        self.commit_weight = commit_weight
        self.vq_weight = vq_weight
        self.ema = False
        self.phn_attr = None
        self.proj_attr = None
        self.use_phn_attr = False
        
        # Latent embedding
        self.embedding = nn.Embedding(vocab_size, self.latent_dim)
        self.onehot = nn.Embedding.from_pretrained(torch.eye(vocab_size), freeze=True)
        #self.register_buffer('mask_pad', torch.arange(vocab_size)==0)
        
        # Scaling factor
        if temp <0:
            self.temp = nn.Parameter(torch.FloatTensor([1]))
        else:
            self.register_buffer('temp', torch.FloatTensor([temp]))
        
        # Criterion for deriving distribution
        self.measurement = neg_batch_l2

    def load_pretrained_embedding(self, old_emb):
        if 'emb.embedding.weight' in old_emb.keys():
            self.embedding = self.embedding.from_pretrained(old_emb['emb.embedding.weight'].data, freeze=False)
            if 'emb.temp' in old_emb.keys(): self.temp.data = old_emb['emb.temp'].data
            if 'emb.running_tok_freq' in old_emb.keys(): self.running_tok_freq = old_emb['emb.running_tok_freq']
            if 'emb.running_ema' in old_emb.keys(): self.terunning_emamp = old_emb['emb.running_ema']
        else:
            self.embedding = self.embedding.from_pretrained(old_emb['emb.weight'], freeze=False)

    def create_msg(self):
        return '           | EMA update = {}\t | Temp. = {}\t| Phn. attributs = {} ( projected = {})'\
            .format(self.ema, 
                    'learnable' if type(self.temp) is nn.Parameter else self.temp.data.item(), 
                    self.use_phn_attr, 
                    self.proj_attr is not None)

class L2Embedding(BaseEmbedding):
    """docstring for DualEmbedding"""
    def __init__(self, vocab_size, ema, softmax, latent_dim, commit_weight, vq_weight, temp, 
                 skip_prob, stop_grad, phn_attr_pth=None, proj_attr=None):
        super().__init__(vocab_size, softmax, latent_dim, commit_weight, vq_weight, temp)
        del self.embedding
        assert self.softmax=='normal'
        assert not ema
        assert commit_weight == 0
        assert vq_weight == 0

        # Skip connection of enc/dec
        self.skip_prob = skip_prob

        # Speech2speech gradient applied on embedding
        self.stop_grad = stop_grad
        
        # Load phn attr (freeze weight plz)
        self.use_phn_attr = phn_attr_pth is not None and phn_attr_pth != ''
        if self.use_phn_attr:
            assert latent_dim>proj_attr>0, 'Currently, proj attr is necessary'
            phn_attr = torch.FloatTensor(read_phn_attr(phn_attr_pth))
            attr_dim = phn_attr.shape[1]
            self.phn_attr = nn.Embedding.from_pretrained(phn_attr, freeze=True, padding_idx=0)
            self.proj_attr = nn.Linear(attr_dim,proj_attr)
        
        # Random init. learnable embedding
        randon_init_dim = latent_dim - proj_attr if self.use_phn_attr else latent_dim
        self.learnable_table = torch.nn.Parameter(torch.randn((vocab_size,randon_init_dim)))
        
    @property
    def embedding(self):
        if self.use_phn_attr:
            full_table = torch.cat([self.learnable_table,
                                    self.proj_attr(self.phn_attr.weight)],dim=-1)
        else:
            full_table = self.learnable_table
        return nn.Embedding.from_pretrained(full_table)

    def inference(self, txt):
        learn_emb = F.embedding(txt, self.learnable_table)
        
        if self.use_phn_attr:
            phn_emb = self.proj_attr(self.phn_attr(txt))
            return torch.cat([learn_emb,phn_emb],dim=-1)
        else:
            return learn_emb

    def forward(self, enc_embs, first_n_real_mel=0):
        B,S,_ = enc_embs.shape

        # Get full embedding table
        if self.use_phn_attr:
            embedding_table = torch.cat([self.learnable_table, self.proj_attr(self.phn_attr.weight)],dim=-1)
        else:
            embedding_table = self.learnable_table

        # Dont let objectives on p_code affect embedding table if input isn't real mel
        if first_n_real_mel>0:
            real_part = F.relu(self.temp)*self.measurement(enc_embs[:first_n_real_mel],
                                                           embedding_table,
                                                           first_n_real_mel,S)
            fake_part = F.relu(self.temp)*self.measurement(enc_embs[first_n_real_mel:],
                                                           embedding_table.detach(),
                                                           B-first_n_real_mel,S)
            similarity = torch.cat([real_part,fake_part],dim=0)
        else:
            similarity = F.relu(self.temp)*self.measurement(enc_embs,embedding_table,B,S)
        
        # Compute enc. output distribution over codebook (based on L2)
        p_code = similarity.softmax(dim=-1)
        
        # Select nearest neighbor in codebook
        picked_idx = p_code.argmax(dim=-1)
        
        if self.stop_grad:
            # Stop-grad version
            picked_code = F.embedding(picked_idx,embedding_table)
        else:
            # ST-onehot version
            p_hard = p_code + (self.onehot(picked_idx) - p_code).detach()
            picked_code = F.linear(p_hard,embedding_table.T)
        
        if self.training and self.skip_prob>0 and np.random.rand()<self.skip_prob:
            # skip connection (only when training)
            new_latent = enc_embs
        else:
            # Quantize
            new_latent = enc_embs + picked_code - enc_embs.detach()

        return p_code, new_latent, 0, 0


class SeperateEmbedding(BaseEmbedding):
    """ Seperate embedding for ASR and TTS (i.e. Speech chain) """
    def __init__(self, vocab_size, ema, softmax, latent_dim, commit_weight, vq_weight, temp, 
                 skip_prob, stop_grad, phn_attr_pth=None, proj_attr=None):
        super().__init__(vocab_size, softmax, latent_dim, commit_weight, vq_weight, temp)
        assert self.softmax=='normal'
        assert not ema
        assert commit_weight == 0
        assert vq_weight == 0
        assert skip_prob == 0
        del self.embedding

        # Stop grad == True for ST-speech chain
        self.stop_grad = stop_grad
        
        # Linear for ASR / embedding for TTS
        self.asr_final_layer = nn.Linear(latent_dim,vocab_size)

        # Load phn attr (freeze weight plz)
        self.use_phn_attr = phn_attr_pth is not None and phn_attr_pth != ''
        if self.use_phn_attr:
            assert latent_dim>proj_attr>0, 'Currently, proj attr is necessary'
            phn_attr = torch.FloatTensor(read_phn_attr(phn_attr_pth))
            attr_dim = phn_attr.shape[1]
            self.phn_attr = nn.Embedding.from_pretrained(phn_attr, freeze=True, padding_idx=0)
            self.proj_attr = nn.Linear(attr_dim,proj_attr)
        else:
            proj_attr = 0
        self.embedding = nn.Embedding(vocab_size,latent_dim-proj_attr)

    def inference(self, txt):
        emb = self.embedding(txt)
        if self.use_phn_attr:
            return torch.cat([emb,self.proj_attr(self.phn_attr(txt))],dim=-1)
        else:
            return emb

    def forward(self, enc_embs, first_n_real_mel=0):
        # first_n_real_mel is redundant here since embeddings are seperated
        # ASR output
        p_code = torch.softmax(self.asr_final_layer(enc_embs),dim=-1)

        # TTS input
        picked_idx = p_code.argmax(dim=-1)
        if self.stop_grad:
            new_latent = self.embedding(picked_idx)
            if self.use_phn_attr:
                new_latent = torch.cat([new_latent,self.proj_attr(self.phn_attr(picked_idx))],dim=-1)
        else:
            p_hard = p_code + (self.onehot(picked_idx) - p_code).detach()
            new_latent = F.linear(p_hard,self.embedding.weight.T)
            if self.use_phn_attr:
                new_latent = torch.cat([new_latent,
                                        self.proj_attr(F.linear(p_hard,self.phn_attr.weight.T))],dim=-1)

        return p_code, new_latent, 0, 0


def neg_batch_l2(x,y,B,S):
    flat_x = x.reshape(B*S,-1) 
    l2_distance =   torch.sum(flat_x.pow(2), dim=-1,keepdim=True) \
                  + torch.sum(y.pow(2), dim=-1)\
                  - 2 * torch.matmul(flat_x, y.t())
    return - l2_distance.view(B,S,-1)
