import torch.nn as nn
from src.module import ConvLayer, MLP

#--- CNN based speech encoder ---#
class CTC(nn.Module):
    def __init__(self, in_dim, out_dim, dim, dropout, kernel, stride, residual, batch_norm, activation,
                 rnn_layers, rnn_dim, rnn_bid, layer_norm):
        super(CTC, self).__init__()

        # Parse config
        self.kernel = kernel
        self.layers = len(self.kernel)
        self.stride = stride
        self.residual = residual
        self.dim = [dim]*self.layers if type(dim) is int else dim
        self.dim = [in_dim] + self.dim # insert input feature dim
        self.rnn_dim = rnn_dim
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.out_dim = out_dim
        self.dropout = dropout
        self.time_reduce_factor = 2**sum([1 for s in stride if s != 1])
        
        # CNN layers
        for l in range(self.layers):
            setattr(self,'layer'+str(l), ConvLayer(in_dim=self.dim[l], out_dim=self.dim[l+1], stride=self.stride[l],
                                                   kernel_size=self.kernel[l], residual=self.residual[l], 
                                                   batch_norm=batch_norm, activation=activation, dropout=dropout))
            #if self.batch_norm:
            #   setattr(self,'batch_norm'+str(l), nn.BatchNorm1d(self.dim[l+1]))
        cur_outdim = self.dim[-1]

        # RNN
        assert self.rnn_dim>0
        self.rnn = nn.LSTM(cur_outdim,rnn_dim,num_layers=rnn_layers,dropout=dropout,
                           bidirectional=rnn_bid, batch_first=True)
        cur_outdim = rnn_dim*2 if rnn_bid else rnn_dim
        if self.layer_norm:
            self.norm_layer = nn.LayerNorm(cur_outdim)
        
        # Proj
        self.drop = nn.Dropout(dropout)
        self.postnet = nn.Linear(cur_outdim, self.out_dim)
        

    def forward(self,  x):
        # BxTxD -> BxDxT
        x = x.permute(0,2,1)

        # CNN feature extraction
        for l in range(self.layers):
            x = getattr(self,'layer'+str(l))(x)
        
        # BxDxT -> BxTxD
        x = x.permute(0,2,1)
        x,_ = self.rnn(x)

        if self.layer_norm:
            x = self.norm_layer(x)
        
        # project to embedding space
        x = self.postnet(self.drop(x))

        return x

#--- PostNet for ASR ---#
class ASRPostnet(nn.Module):
    def __init__(self, latent_dim, vocab_size):
        super().__init__()

        self.rnn = nn.LSTM(latent_dim,latent_dim,num_layers=2,dropout=0.5,
                           bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(latent_dim*2,vocab_size)
        
    def forward(self,  x):
        # BxTxD -> BxDxT
        x,_ = self.rnn(x)
        x = self.linear(self.dropout(x))
        return x.log_softmax(dim=-1)
