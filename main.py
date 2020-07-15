#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import random
import argparse
import numpy as np

# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--njobs', default=5, type=int, help='Number of threads for decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--debug', action='store_true', help='Debug use.')
parser.add_argument('--no-pin', action='store_true', help='Disable pin-memory for dataloader')
#parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--asr-decode', action='store_true', help='ASR beam decode using Tensorflow.')
parser.add_argument('--gen-specgram', action='store_true', help='Generating mel/linear spectrogram.')
parser.add_argument('--gen-gt-specgram', action='store_true', help='Generating mel/linear spectrogram.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--actual-len', action='store_true', help='Using actual len for CTC loss.')
parser.add_argument('--store-best-per', action='store_true', help='Only store the model with best PER.')
parser.add_argument('--asr-only', action='store_true', help='Only train supervised ASR.')
parser.add_argument('--gen-wav', action='store_true', help='Generate waveform using Griffin-Lim.')
#parser.add_argument('--pretrain_speech', action='store_true', help='Pretrain mode for ASR-decoder.') # ToDo
#parser.add_argument('--pretrain_text', action='store_true', help='Pretrain mode for TTS-decoder.')  # ToDo
#parser.add_argument('--train_tts', action='store_true', help='Test functionality of TTS.')  # ToDo
#parser.add_argument('--train_semi_tts', action='store_true', help='Train semi-TTS.')  # ToDo
paras = parser.parse_args()
setattr(paras,'gpu',not paras.cpu)
setattr(paras,'pin_memory', False if paras.cpu else paras.no_pin)
setattr(paras,'verbose',not paras.no_msg)
config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

if paras.asr_decode:
    mode = 'test'
    from bin.asr_decode import VqvaeDecoder as Solver
elif paras.gen_specgram:
    mode = 'test'
    from bin.gen_specgram import SpecgramGenerator as Solver
elif paras.gen_gt_specgram:
    mode = 'test'
    from bin.gen_gt_specgram import SpecgramGenerator as Solver
elif paras.asr_only:
    mode = 'train'
    from bin.train_asr import VqvaeTrainer as Solver
else:
    mode = 'train'
    from bin.train_vqvae import VqvaeTrainer as Solver

solver = Solver(config,paras,mode)
solver.load_data()
solver.set_model()
solver.exec()
