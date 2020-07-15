import os
import yaml
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import soundfile as sf
from os.path import join

from src.audio import load_audio_transform

LIST = [
	'LJ010-0057-spec.npy',
	'LJ027-0067-spec.npy',
	'LJ009-0213-spec.npy',
	'LJ034-0190-spec.npy',
	'LJ005-0281-spec.npy',
	'LJ002-0054-spec.npy',
	'LJ028-0259-spec.npy',
	'LJ012-0022-spec.npy',
	'LJ006-0039-spec.npy',
	'LJ019-0060-spec.npy',
	'LJ023-0001-spec.npy',
	'LJ044-0108-spec.npy',
	'LJ007-0219-spec.npy',
	'LJ016-0258-spec.npy',
	'LJ042-0113-spec.npy',
	'LJ013-0087-spec.npy',
	'LJ010-0003-spec.npy',
	'LJ019-0128-spec.npy',
	'LJ013-0200-spec.npy',
	'LJ001-0131-spec.npy'
]


def run(paras):
	os.makedirs(paras.output_dir, exist_ok=True)
	config = yaml.load(open(paras.config), Loader=yaml.FullLoader)
	audio_converter = load_audio_transform(**config['data']['audio'])
	files = sorted(glob(join(paras.specgram_dir, '*-spec.npy')))
	if paras.sample:
		files = [f for f in files if f.split('/')[-1] in LIST]
	for f in tqdm(files):
	    specgram = torch.from_numpy(np.load(f))
	    wav, sr = audio_converter.feat_to_wave(specgram)
	    sf.write(join(paras.output_dir, f.split('/')[-1].replace('-spec.npy', '.wav')), wav, sr)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert spectrogram into raw waveform.')
	parser.add_argument('--config', type=str, help='Path to experiment config.')
	parser.add_argument('--specgram-dir', type=str, help='Path to input spectrogram.')
	parser.add_argument('--output-dir', type=str, help='Path to output wave.')
	parser.add_argument('--sample', action='store_true', help='Only sample somes wavs.')
	paras = parser.parse_args()
	run(paras)



