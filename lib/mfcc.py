import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

DELTA_ORDER = 2
DELTA_WINDOW_SIZE = 2
APPLY_CMVN = True

class CMVN(torch.jit.ScriptModule):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


class Delta(torch.jit.ScriptModule):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


class Postprocess(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        # [channel, feature_dim, time] -> [time, channel, feature_dim]
        x = x.permute(2, 0, 1)
        # [time, channel, feature_dim] -> [time, feature_dim * channel]
        return x.reshape(x.size(0), -1).detach()


class ExtractMFCC(nn.Module):
    def __init__(self, sr):
        super(ExtractMFCC, self).__init__()
        self.sr = sr
        self.extract_fn = torchaudio.compliance.kaldi.mfcc
    def forward(self, waveform):
        y = self.extract_fn(waveform, sample_frequency=self.sr, frame_length=50.0, frame_shift=12.5)
        return y.transpose(0, 1).unsqueeze(0)



def create_mfcc_transform(sample_rate):

    transforms = [ExtractMFCC(sample_rate)]

    if DELTA_ORDER >= 1:
        transforms.append(Delta(DELTA_ORDER, DELTA_WINDOW_SIZE))

    if APPLY_CMVN:
        transforms.append(CMVN())

    transforms.append(Postprocess())

    mfcc_dim = 13*(DELTA_ORDER+1)

    return nn.Sequential(*transforms), mfcc_dim