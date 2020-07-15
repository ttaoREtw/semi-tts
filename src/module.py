import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from src.util import get_audio_feat_mask

MAX_SEQ_LEN = 5500 # for postion encoding

# --------
# For spkr
# --------
class SpeakerEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, filters, dropout):
        super(SpeakerEncoder, self).__init__()
        in_size = [in_dim] + filters
        out_size = filters + [out_dim//2]
        act_fn = ['relu'] * (len(out_size) - 1) + ['linear']
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1d(
                    in_channels=Din, 
                    out_channels=Dout,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain=fn),
                nn.BatchNorm1d(Dout),
                nn.ReLU() if fn == 'relu' else nn.Identity(),
                nn.Dropout(dropout)) 
            for Din, Dout, fn in zip(in_size, out_size, act_fn)
        ])

    def forward(self, x):
        """
        Args:
            x: shape of (batch size, frames, n_mels)
        Return:
            concatenated hidden mean and hidden std
        """
        x = x.transpose(1, 2)
        for layer in self.convs:
            x = layer(x)
        x = x.transpose(1, 2)
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        return torch.cat([mean, std], dim=-1)


# --------
# For tts
# --------
class Postnet(nn.Module):
    """Postnet for tacotron2"""
    def __init__(self, n_mels, postnet_embed_dim, postnet_kernel_size, postnet_n_conv, postnet_dropout):
        super(Postnet, self).__init__()

        in_size = [n_mels] + [postnet_embed_dim] * (postnet_n_conv-1) 
        out_size = [postnet_embed_dim] * (postnet_n_conv-1) + [n_mels]
        act_fn = ['tanh'] * (postnet_n_conv-1) + ['linear']
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1d(
                    in_channels=Din, 
                    out_channels=Dout,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=(postnet_kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain=fn),
                nn.BatchNorm1d(Dout),
                nn.Tanh() if fn == 'tanh' else nn.Identity(),
                nn.Dropout(postnet_dropout)) 
            for Din, Dout, fn in zip(in_size, out_size, act_fn)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        return x 


class Decoder(nn.Module):
    """Tacotron2 decoder consists of prenet, attention and LSTM"""
    def __init__(self, n_mels, n_frames_per_step, enc_embed_dim, spkr_embed_dim, prenet_dim, prenet_dropout,
                 query_rnn_dim, dec_rnn_dim, query_dropout, dec_dropout, 
                 attn_dim, n_location_filters, location_kernel_size, loc_aware, 
                 use_summed_weights, drop_dec_in, prenet_norm_type=None, pretrain=False, spkr_embed_mode='adaIN'):
        super(Decoder, self).__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.enc_embed_dim = enc_embed_dim
        self.spkr_embed_dim = spkr_embed_dim
        self.query_rnn_dim = query_rnn_dim
        self.dec_rnn_dim = dec_rnn_dim
        self.prenet_dropout = prenet_dropout
        self.prenet_dim = prenet_dim
        self.query_dropout = nn.Dropout(query_dropout)
        self.dec_dropout = nn.Dropout(dec_dropout)
        self.attn_dim = attn_dim
        self.n_location_filters = n_location_filters
        self.location_kernel_size = location_kernel_size
        self.pretrain = pretrain
        self.loc_aware = loc_aware
        self.use_summed_weights = use_summed_weights
        self.drop_dec_in = drop_dec_in
        self.prenet_norm_type = prenet_norm_type
        self.spkr_embed_mode = spkr_embed_mode.lower()
        if self.spkr_embed_mode == 'adain':
            self.pseudo_latent_mean = nn.Linear(self.spkr_embed_dim, self.query_rnn_dim)
            self.pseudo_latent_std = nn.Sequential(
                                        nn.Linear(self.spkr_embed_dim, self.query_rnn_dim),
                                        nn.ReLU())
        elif self.spkr_embed_mode == 'concat':
            self.spkr_mem_proj = nn.Linear(self.spkr_embed_dim + self.enc_embed_dim, self.enc_embed_dim)
        elif self.spkr_embed_mode == 'add':
            self.spkr_proj = nn.Linear(self.spkr_embed_dim, self.enc_embed_dim)
            self.spkr_mem_proj = nn.Linear(self.enc_embed_dim, self.enc_embed_dim)
        else:
            raise NotImplementedError

        self.prenet = Prenet(
            n_mels * n_frames_per_step, [prenet_dim, prenet_dim], 
            apply_dropout=prenet_dropout, norm_type=prenet_norm_type)
        self.query_rnn = nn.LSTMCell(
            prenet_dim + enc_embed_dim, query_rnn_dim)
        self.attn = Attention(
            query_rnn_dim, enc_embed_dim, attn_dim, 
            n_location_filters, location_kernel_size,
            loc_aware, use_summed_weights)
        self.dec_rnn = nn.LSTMCell(
            query_rnn_dim + enc_embed_dim, dec_rnn_dim)
        self.proj = Linear(
            dec_rnn_dim + enc_embed_dim, n_mels * n_frames_per_step)
        self.gate_layer = Linear(
            dec_rnn_dim + enc_embed_dim, 1, bias=True, w_init_gain='sigmoid')

    def forward(self, memory, memory_lengths, teacher, spkr_embed, tf_rate=0.0, unpair_max_frame=None):
        """
        Arg:
            decoder_inputs: melspectrogram of shape (B, T, n_mels), None if inference
            memory: encoder outputs (B, L, D). It could be None if pretraining
            memory_lengths: the lengths of memory without padding, it could be None if pretraining
            teacher: melspectrogram provided as teacher or int. serving as max_dec_step

        Return:
            mel_outputs: (B, T, n_mels)
            alignments: (B, T // n_frames_per_step, L)
            
        """
        # Init.
        device = memory.device
        B = memory.size(0)
        if type(teacher) is not int: # if training now
            # text2text only happen during training
            teacher_bs = teacher.shape[0]
            partial_no_teacher = B!=teacher_bs                

        go_frame = torch.zeros( B, self.n_frames_per_step, self.n_mels, device=device)
        self.init_decoder_states(memory)
        mask = None # if self.pretrain else self._make_mask(memory_lengths)

        # Stage check
        inference = tf_rate==0.0
        if inference:
            decode_steps = teacher//self.n_frames_per_step if type(teacher) == int else teacher.shape[1]
        else: # training
            if partial_no_teacher:
                assert unpair_max_frame is not None
                decode_steps = max(teacher.shape[1]//self.n_frames_per_step,
                                   unpair_max_frame//self.n_frames_per_step)

            else:
                # (B, T, n_mels) -> (B, T' (decode steps), n_frames_per_step, n_mels)
                decode_steps = teacher.shape[1]//self.n_frames_per_step
            teacher = teacher.view(teacher.shape[0], -1, self.n_mels*self.n_frames_per_step)
            teacher = self.prenet(teacher)

        # Forward
        mel_outputs, alignments, stops = [], [], []
        dec_in = self.prenet(go_frame.view(B, -1))
        for t in range(decode_steps):
            mel_out, align, stop = self.decode_one_step(dec_in, spkr_embed, mask)
            mel_outputs.append(mel_out)
            alignments.append(align)
            stops.append(stop)

            if inference or (np.random.rand()>tf_rate):
                # Use previous output
                dec_in = self.prenet(mel_out.view(B, self.n_frames_per_step * self.n_mels))
            elif np.random.rand()<self.drop_dec_in:
                dec_in = teacher.mean(dim=1)
                # Unpaired text
                if partial_no_teacher:
                    unpaired_prev_out = mel_out[teacher_bs:].view(-1, self.n_frames_per_step * self.n_mels).contiguous()
                    dec_in = torch.cat([dec_in,self.prenet(unpaired_prev_out)],dim=0)
            else:
                # Use ground truth
                take_frame = min(t,teacher.shape[1]-1)
                dec_in = teacher[:, take_frame, :]
                # Unpaired text
                if partial_no_teacher:
                    unpaired_prev_out = mel_out[teacher_bs:].view(-1, self.n_frames_per_step * self.n_mels).contiguous()
                    dec_in = torch.cat([dec_in,self.prenet(unpaired_prev_out)],dim=0)

        # (B, T, n_mels)
        mel_outputs = torch.cat(mel_outputs, dim=1)
        # (B, T', L)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (B, T)
        stops = torch.cat(stops, dim=1)
        return mel_outputs, alignments, stops

    def decode_one_step(self, dec_in, spkr_embed, mask):
        """
        Arg:
            dec_in: melspectrogram of shape (B, n_frames_per_step * prenet_dim)
            mask: None if pretraining
        Return:
            mel_out: (B, n_frames_per_step, n_mels)
            attn_weights: (B, L)
        """
        B = dec_in.size(0)
        # For query_rnn
        query_rnn_input = torch.cat([dec_in, self.attn_context], dim=-1)
        hidden, cell = self.query_rnn(
            query_rnn_input, (self.query_rnn_hidden, self.query_rnn_cell))
        self.query_rnn_hidden = self.query_dropout(hidden)
        self.query_rnn_cell   = cell

        # Attention weights (for location-awared attention)
        # (B, 2, L)
        if self.use_summed_weights:
            attn_history = torch.stack([
                self.attn_weights, self.attn_weights_sum]).transpose(0, 1)
        else:
            attn_history = self.attn_weights.unsqueeze(1)
        # Perform attention
        if self.pretrain:
            ctx = torch.zeros_like(self.attn_context)
            weights = torch.zeros_like(self.attn_weights)
        else:
            mem = self.memory
            if self.spkr_embed_mode == 'concat':
                mem = self.spkr_mem_proj(torch.cat(
                    [mem, spkr_embed.unsqueeze(1).repeat(1, self.memory.size(1), 1)], dim=-1))
            elif self.spkr_embed_mode == 'add':
                mem = self.spkr_mem_proj(mem + self.spkr_proj(spkr_embed.unsqueeze(1)))
            elif self.spkr_embed_mode == 'adain':
                pass
            else:
                raise NotImplementedError

            ctx, weights = self.attn(
                query=self.query_rnn_hidden, 
                memory=mem, 
                processed_memory=self.processed_memory, 
                attn_history=attn_history, 
                mask=mask)
        self.attn_context = ctx
        self.attn_weights = weights
        self.attn_weights_sum = weights + self.attn_weights_sum

        # # Speaker adaption
        if self.spkr_embed_mode == 'adain':
            spkr_adapted_hidden = self.pseudo_latent_std(spkr_embed) *\
                                 (self.query_rnn_hidden - self.pseudo_latent_mean(spkr_embed))
        else:
            # Not adapt
            spkr_adapted_hidden = self.query_rnn_hidden
        
        # For dec_rnn
        dec_rnn_input = torch.cat([
            self.attn_context, spkr_adapted_hidden], dim=-1)
        hidden, cell = self.dec_rnn(
            dec_rnn_input, (self.dec_rnn_hidden, self.dec_rnn_cell))
        self.dec_rnn_hidden = self.dec_dropout(hidden)
        self.dec_rnn_cell   = cell
        # To predict mel output
        dec_rnn_hidden_attn_context = torch.cat([
            self.dec_rnn_hidden, 
            self.attn_context], dim=-1)
        mel_out = self.proj(dec_rnn_hidden_attn_context)
        mel_out = mel_out.view(B, self.n_frames_per_step, self.n_mels)
        stop = self.gate_layer(dec_rnn_hidden_attn_context).repeat(1, self.n_frames_per_step)
        return mel_out, self.attn_weights, stop
    
    def init_decoder_states(self, memory):
        B = memory.size(0)
        L = memory.size(1)
        device = memory.device
        # RNN states
        self.query_rnn_hidden = torch.zeros((B, self.query_rnn_dim), requires_grad=True, device=device)
        self.query_rnn_cell   = torch.zeros((B, self.query_rnn_dim), requires_grad=True, device=device)
        self.dec_rnn_hidden = torch.zeros((B, self.dec_rnn_dim), requires_grad=True, device=device)
        self.dec_rnn_cell   = torch.zeros((B, self.dec_rnn_dim), requires_grad=True, device=device)
        # Attention weights
        self.attn_weights     = torch.zeros((B, L), requires_grad=True, device=device)
        self.attn_weights_sum = torch.zeros((B, L), requires_grad=True, device=device)
        # Attention context
        self.attn_context = torch.zeros((B, self.enc_embed_dim), requires_grad=True, device=device)
        # Encoder output
        self.memory = memory
        self.processed_memory = self.attn.memory_layer(memory)

    @staticmethod
    def _make_mask(lengths):
        """
        Return:
            mask with 1 for not padded part and 0 for padded part
        """
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).to(lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return ~mask
    
# https://github.com/mozilla/TTS/issues/26
class Prenet(nn.Module):
    def __init__(self, in_dim, hidden_dim=[256, 256], apply_dropout=0.5, norm_type=None):
        super(Prenet, self).__init__()
        input_dim = [in_dim] + hidden_dim[:-1]
        self.layers = nn.ModuleList([
            Linear(Din, Dout, bias=False, norm_type=norm_type) 
            for Din, Dout in zip(input_dim, hidden_dim)])
        self.relu = nn.ReLU()
        self.apply_dropout = apply_dropout
        
    def forward(self, x):
        """
        Arg:
            x: part of melspectrogram (batch, ..., n_frames_per_step * n_mels)
        Return:
            A tensor of shape (batch, ..., prenet_dim)
        """
        for layer in self.layers:
            # The dropout does NOT turn off even when doing evaluation.
            x = F.dropout(self.relu(layer(x)), p=self.apply_dropout, training=True)
        return x
    
    
class Attention(nn.Module):
    """Atention module"""
    def __init__(self, query_dim, memory_dim, hidden_dim, 
                 n_location_filters, location_kernel_size, 
                 loc_aware, use_summed_weights):
        super(Attention, self).__init__()
        self.query_layer = Linear(query_dim, hidden_dim, 
                                  bias=False, w_init_gain='tanh')
        self.memory_layer = Linear(memory_dim, hidden_dim, 
                                   bias=False, w_init_gain='tanh')
        self.v = Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        # loc: location-awared part
        self.loc_aware = loc_aware
        self.use_summed_weights = use_summed_weights
        if loc_aware:
            in_channels = 2 if use_summed_weights else 1
            self.loc_conv = Conv1d(in_channels=in_channels, 
                                   out_channels=n_location_filters, 
                                   kernel_size=location_kernel_size, 
                                   bias=False, stride=1, dilation=1)
            self.loc_linear = Linear(n_location_filters, hidden_dim, 
                                     bias=False, w_init_gain='tanh')
        
    def process_memory(self, memory):
        # Don't need to process memory for each decode timestep
        return self.memory_layer(memory)
    
    def energy(self, query, processed_memory, attn_history):
        """
        Arg:
            query: (B, out_dim of `prenet`)
            processed_memory: (B, L, hidden_dim)
            attn_history: (B, 2, L)
        Return:
            energy of shape (B, L)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        
        # Get location-awared feature
        if self.loc_aware:
            processed_loc_feat = self.loc_conv(attn_history).transpose(1, 2)
            processed_loc_feat = self.loc_linear(processed_loc_feat)
        else:
            processed_loc_feat = 0
        # Calculate energy
        energy = self.v(self.tanh(
            processed_query + processed_loc_feat + processed_memory))
        energy = energy.squeeze(-1)
        return energy
        
    def forward(self, query, memory, processed_memory, 
                attn_history, mask):
        energy = self.energy(query, processed_memory, attn_history)
        
        # Fill the -inf for padding encoder output
        if mask is not None:
            energy.data.masked_fill_(mask, -float('inf'))
    
        # (B, L)
        attn_weights = F.softmax(energy, dim=1)
        # Broadcast bmm, (B, 1, hidden_dim)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), memory)
        attn_context = attn_context.squeeze(1)
        return attn_context, attn_weights
    

class Encoder(nn.Module):
    """Tacotron2 text encoder consists of convolution layers and bidirectional LSTM
    """
    def __init__(self, in_dim, enc_embed_dim, enc_n_conv, enc_rnn_layer, enc_kernel_size, enc_dropout=0.5):
        super(Encoder, self).__init__()
        in_size = [in_dim] + [enc_embed_dim] * (enc_n_conv-1) 
        out_size = [enc_embed_dim] * enc_n_conv
        self.enc_embed_dim = enc_embed_dim
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1d(in_channels=Din, 
                   out_channels=Dout,
                   kernel_size=enc_kernel_size,
                   stride=1,
                   padding=(enc_kernel_size - 1) // 2,
                   dilation=1,
                   w_init_gain='relu'),
                nn.BatchNorm1d(enc_embed_dim), 
                nn.ReLU(),
                nn.Dropout(enc_dropout)) 
            for Din, Dout in zip(in_size, out_size)
        ])
        self.lstm = nn.LSTM(
            input_size=enc_embed_dim, 
            hidden_size=enc_embed_dim // 2,
            num_layers=enc_rnn_layer,
            batch_first=True,
            bidirectional=True
        )        
        
    def forward(self, txt_embed, input_lengths):
        """
        Arg:
            txt_embed: torch.FloatTensor of shape (batch, L, enc_embed_dim)
        Return:
            The hidden representation of text, (batch, L, enc_embed_dim)
        """
        # (B, D, L)
        x = txt_embed.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        # (B, L, D)
        x = x.transpose(1, 2)
        
        #input_lengths = input_lengths.cpu().numpy()
        #x = nn.utils.rnn.pack_padded_sequence(
        #    x, input_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        
        output, _ = self.lstm(x)
        #output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output
        
    
class TextEmbedding(nn.Module):
    def __init__(self, n_vocab, embedding_size):
        super(TextEmbedding, self).__init__()
        self.embed = nn.Embedding(n_vocab, embedding_size)
        
    def forward(self, text):
        """
        Arg:
            text: torch.LongTensor of shape (batch, length)
        Return:
            Embedding of shape (batch, length, embedding_size)
        """
        return self.embed(text)

    
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv1d, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1), "kernel_size should be odd if no given padding."
            padding = (dilation * (kernel_size - 1)) // 2

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            bias=bias)
        
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        
    def forward(self, x):
        return self.conv(x)
        
        
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear', norm_type=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

        self.apply_norm = norm_type is not None
        if self.apply_norm:
            assert norm_type in ['LayerNorm','BatchNorm1d']
            self.norm_type = norm_type
            self.norm = getattr(nn,norm_type)(out_dim)#nn.BatchNorm1d(out_dim)
    
    def forward(self, x):
        x = self.linear(x)
        if self.apply_norm:
            if len(x.shape)==3 and self.norm_type=='BatchNorm1d':
                x = x.transpose(1,2)
            x = self.norm(x)
            if len(x.shape)==3 and self.norm_type=='BatchNorm1d':
                x = x.transpose(1,2)
        return x
    
### ------------------------------------------- 
### -------------- For CBHG -------------------
### ------------------------------------------- 
class BatchNormConv1d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_size, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.bn(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H = self.relu(self.H(x))
        T = self.sigmoid(self.T(x))
        y = H * T + x * (1.0 - T)
        return y


class CBHG(nn.Module):
    """CBHG in original paper.
    Components:
        - 1-d convolution banks
        - highway networks
        - gru (bidirectional)
    """
    def __init__(self, in_dim, K=16, hidden_sizes=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
                [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                    padding=k//2, activation=self.relu)
                for k in range(1, K+1)])
        self.pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + hidden_sizes[:-1]
        activations = [self.relu] * (len(hidden_sizes) - 1) + [None]
        self.conv1d_projs = nn.ModuleList(
                [BatchNormConv1d(in_size, out_size, kernel_size=3,
                    stride=1, padding=1, activation=act)
                    for in_size, out_size, act in zip(in_sizes, hidden_sizes, activations)])

        self.pre_highway_proj = nn.Linear(hidden_sizes[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
                [Highway(in_dim, in_dim) for _ in range(4)])
        self.gru = nn.GRU(
                in_dim, in_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        x = inputs
        # Assert x's shape: (batch_size, timesteps, in_dim)
        assert x.size(-1) == self.in_dim
        # -> (batch_size, in_dim, timesteps)
        x = x.transpose(1, 2)
        T = x.size(-1)

        # -> (batch_size, in_dim * K, timesteps)
        x = torch.cat(
                [conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projs:
            x = conv1d(x)
        # -> (batch_size, timesteps, hidden_dim)
        x = x.transpose(1, 2)
        # -> (batch_size, timesteps, in_dim)
        x = self.pre_highway_proj(x)

        x = x+inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
        # -> (batch_size, timesteps, 2 * in_dim)
        y, _ = self.gru(x)

        if input_lengths is not None:
            y, _ = nn.utils.rnn.pad_packed_sequence(
                    y, batch_first=True)
        return y

# --------
# For asr
# --------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, residual, batch_norm, activation, dropout):
        super(ConvLayer, self).__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        self.activation = getattr(torch,activation.lower())
        padding = 1 if kernel_size!=1 else 0 # if k = 1, it's a FC with time at last axis
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding=padding)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        feat = self.conv(x)
        if self.batch_norm:
            feat = self.bn(feat)
        feat = self.activation(feat)
        if self.residual:
            feat = feat+x
        feat = self.drop(feat)

        return feat

class MLP(nn.Module):
    def __init__(self, in_dim, dim,relu=True,dropout=0.0):
        super(MLP, self).__init__()
        self.layers = len(dim)
        self.dim = dim
        hid_dims = [in_dim]+dim
        self.out_dim = hid_dims[-1]

        module_list = []
        for l in range(self.layers):
            module_list.append(nn.Linear(hid_dims[l],hid_dims[l+1]))
            if relu:
                module_list.append(nn.ReLU())
            if dropout>0:
                module_list.append(nn.Dropout(dropout))

        self.netwrok = nn.Sequential(*module_list)

    def forward(self, inputs):
        return self.netwrok(inputs)
