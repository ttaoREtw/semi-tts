## Semi-supervised TTS
Semi-supervised Learning for Multi-speaker Text-to-speech Synthesis Using Discrete Speech Representation
<p align="center">
  <img src="illustration.png">
</p>

### Data
#### Prepare data
Download VCTK and LJSpeech and put it into `data/audio-corpus`. Specifically, waves from LJSpeech should be in `data/audio-corpus/lj` and waves from speaker p225 of VCTK should be in `data/audio-corpus/p225`, etc..
The data partition is specified in `data/partition_tables/<partition-table.csv>`.
The phoneme transcription of each wave file is in `data/map_tables/lj_vctk_g2p.csv`.
>Note that for members in **NTU speech lab**, the audio-corpus could be download from `/groups/public/ttao/audio-corpus.zip`.

#### Audio preprocessing
The hyperparameter for audio features could be modified in `config/<config.yaml>`.
The audio preprocessing code is in `src/audio.py`.

### Model
To adjust the model hyperparameters or learning rate, please modify the configure file in `config/<config.yaml>`.

### Running
#### Train from scratch
```sh
python main.py --config config/<config.yaml> --njobs <num-workers>
```
#### Continue training
```sh
python main.py --config config/<config.yaml>\
               --njobs <num-workers>\
               --load <checkpoint-path>
```
The training log could be found in directory `log/`.
>Note that for members in **NTU speech lab**, the checkpoints could be download from `/groups/public/ttao/semi-tts-ckpt.zip`.

#### Inference
```sh
python main.py --gen-specgram\
               --config config/<config.yaml>\
               --njobs <num-workers>\
               --load <checkpoint-path>\
               --logdir <output-directory>
```

### Reference
* Grapheme-to-phoneme tool: https://github.com/Kyubyong/g2p
* Tacotron-2: https://github.com/NVIDIA/tacotron2
* End-to-end ASR: https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch

### Citation
```
@inproceedings{liu2020towards,
  title={Towards unsupervised speech recognition and synthesis with quantized speech representation learning},
  author={Liu, Alexander H and Tu, Tao and Lee, Hung-yi and Lee, Lin-shan},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7259--7263},
  year={2020},
  organization={IEEE}
}

@article{tu2020semi,
  title={Semi-supervised Learning for Multi-speaker Text-to-speech Synthesis Using Discrete Speech Representation},
  author={Tu, Tao and Chen, Yuan-Jui and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2005.08024},
  year={2020}
}
```
