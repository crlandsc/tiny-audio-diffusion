<div align="center">
    <img src="./images/CL Banner.png"/>
</div>

<br />

![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face_Spaces-blue) ![Static Badge](https://img.shields.io/badge/Repo_Tutorial-red?logo=YouTube) ![Static Badge](https://img.shields.io/badge/Medium-red?logo=Medium&color=black) ![GitHub Repo stars](https://img.shields.io/github/stars/crlandsc/tiny-audio-diffusion?color=gold) ![GitHub forks](https://img.shields.io/github/forks/crlandsc/tiny-audio-diffusion?color=green)


This is a repository for generating short audio samples and training waveform diffusion models on a GPU with less than 2GB VRAM.

## Motivation

The purpose of this project is to provide access to stereo high-resolution (44.1kHz) conditional and unconditional audio waveform (1D U-Net) diffusion code for those interested in exploration but who have limited resources. There are many methods for audio generation on low-level hardware, but less so specifically for waveform-based diffusion.

The repository is built heavily adapting code from Archinet's [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch) library. A huge thank you to [Flavio Schneider](https://github.com/flavioschneider) for his incredible open-source work in this field!


## Background

Direct waveform diffusion is inherently computationally intensive. For example, an audio sample with the industry standard 44.1kHz sampling rate requires 44,100 samples for just 1 second of audio. Now multiply that by 2 for a stereo file. However, it has a significant advantage over many methods that reduce audio into spectrograms or downsample - the network retains and learns from *phase* information. Phase is challenging to represent on its own in visual methods, such as spectrograms, as it appears similar to that of random noise. Because of this, many generative methods discard phase information and then implement ways of estimating and regenerating it. However, it plays a key role in defining the timbral qualities of sounds and should not be dispensed with so easily.

Waveform diffusion is able to retain this important feature as it does not perform any transforms on the audio before feeding it into the network. This is how humans perceive sounds, with both amplitude and phase information bundled together in a single signal. As mentioned previously, this comes at the expense of computational requirements and is often reserved for training on a cluster of GPUs with high speeds and lots of memory. Because of this, it is hard to begin to experiment with waveform diffusion with limited resources.

This repository seeks to offer some base code to those looking to experiment with and learn more about waveform diffusion on their own computer without having to purchase cloud resources or upgrade hardware. This goes for not only *inference*, but *training* your own models as well!

To make this feasible, however, there must be a tradeoff of quality, speed, and sample length. Because of this, I have focused on training base models for one-shot drum samples - as they are inherently short in sample length.

The current configuration is set up to be able to train ~0.75 second stereo samples at 44.1kHz, allowing for the generation of high-quality one-shot audio samples. The network configuration can be adjusted to improve the resolution, sample rate, training and inference speed, sample length, etc. but, of course, more hardware resources will be required.

Other methods of diffusion, such as diffusion in the latent space ([Stable Diffusion's](https://stability.ai/stablediffusion) secret sauce), compared to this repo's raw waveform diffusion can offer an improvement and other tradeoffs between quality, memory requirements, speed, etc. I recommend this repo to remain up-to-date with the latest research in generative audio: https://github.com/archinetai/audio-ai-timeline

Also recommended is [Harmonai's](https://www.harmonai.org/) community project, [Dance Diffusion](https://github.com/Harmonai-org/sample-generator), which implements similar functionality to this repo on a larger scale with several pre-trained models. [Colab notebook](https://colab.research.google.com/github/Harmonai-org/sample-generator/blob/main/Dance_Diffusion.ipynb) available.

---

## Setup

Follow these steps to set up an environment for both generating audio samples and training models.

*NOTE:* To use this repo with a GPU, you must have a CUDA-capable GPU and have the CUDA toolkit installed for your specific to your system (ex. Linux, x86_64, WSL-Ubuntu). More information can be found [here](https://developer.nvidia.com/cuda-toolkit).

#### 1. Create a Virtual Environment:

Ensure that [Anaconda (or Miniconda)](https://docs.anaconda.com/free/anaconda/install/index.html) is installed and activated. From the command line, `cd` into the [`setup/`](setup/) folder and run the following lines:
```bash
conda env create -f environment.yml
conda activate tiny-audio-diffusion
```

This will create and activate a conda environment from the [`setup/environment.yml`](setup/environment.yml) file and install the dependencies in [`setup/requirements.txt`](setup/requirements.txt).

#### 2. Install Python Kernel For Jupyter Notebook

Run the following line to create a kernel for the current environment to run the inference notebook.

```bash
python -m ipykernel install --user --name tiny-audio-diffusion --display-name "tiny-audio-diffusion (Python 3.10)"
```

#### 3. Define Environment Variables

Rename [`.env.tmp`](.env.tmp) to `.env` and replace the entries with your own variables (example values are random).

```bash
DIR_LOGS=/logs
DIR_DATA=/data

# Required if using Weights & Biases (W&B) logger
WANDB_PROJECT=tiny_drum_diffusion # Custom W&B name for current project
WANDB_ENTITY=johnsmith # W&B username
WANDB_API_KEY=a21dzbqlybbzccqla4txa21dzbqlybbzccqla4tx # W&B API key
```

*NOTE:* Sign up for a [Weights & Biases](https://wandb.ai/site) account to log audio samples, spectrograms, and other metrics while training (it's free!).

W&B logging example for this repo [here](https://wandb.ai/crlandsc/unconditional-drum-diffusion?workspace=user-crlandsc).

---

## Pre-trained Models

Pretrained models can be found on Hugging Face (each model contains a `.ckpt` and `.yaml` file):

|Model|Link|
|---|---|
|Kicks|[crlandsc/tiny-audio-diffusion-kicks](https://huggingface.co/crlandsc/tiny-audio-diffusion-kicks)|
|Snares|[crlandsc/tiny-audio-diffusion-snares](https://huggingface.co/crlandsc/tiny-audio-diffusion-snares)|
|Hi-hats|[crlandsc/tiny-audio-diffusion-hihats](https://huggingface.co/crlandsc/tiny-audio-diffusion-hihats)|
|Percussion (all drum types)|[crlandsc/tiny-audio-diffusion-percussion](https://huggingface.co/crlandsc/tiny-audio-diffusion-percussion)|

*Follow current model training progress [here](https://wandb.ai/crlandsc/unconditional-drum-diffusion?workspace=user-crlandsc) (more models will be added as they are trained).*

Pre-trained models can be downloaded to generate samples via the [inference notebook](Inference.ipynb). They can also be used as a base model to fine-tune on custom data. It is recommended to create subfolders within the [`saved_models`](saved_models/) folder to store each model's `.ckpt` and `.yaml` files.

---

## Inference
### Hugging Face Spaces
Generate samples withot code on [🤗 Hugging Face Spaces](https://huggingface.co/spaces/crlandsc/tiny-audio-diffusion)

### Jupyter Notebook
#### Audio Sample Generation
Current Capabilities:
- Unconditional Generation
- Conditional "Style-transfer" Generation

Open the [`Inference.ipynb`](Inference.ipynb) in Jupyter Notebook and follow the instructions to generate new audio samples. Ensure that the `"tiny-audio-diffusion (Python 3.10)"` kernel is active in Jupyter to run the notebook and you have downloaded the [pre-trained model](#Pre\-trained-Models) of interest from Hugging Face.

---

## Train

The model architecture has been constructed with [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) and [Hydra](https://hydra.cc/docs/intro/) frameworks. All configurations for the model are contained within `.yaml` files and should be edited there rather than hardcoded.

[`exp/drum_diffusion.yaml`](exp/drum_diffusion.yaml) contains the default model configuration. Additional custom model configurations can be added to the [`exp`](exp/) folder.

Custom models can be trained or fine-tuned on custom datasets. Datasets should consist of a folder of `.wav` audio files with a 44.1kHz sampling rate.

To train or finetune models, run one of the following commands in the terminal from the repo's root folder and replace `<path/to/your/train/data>` with the path to your custom training set.


**Train model from scratch (on CPU):**
*(not recommended)*

```bash
python train.py exp=drum_diffusion datamodule.dataset.path=<path/to/your/train/data>
```


**Train model from scratch (on GPU):**

```bash
python train.py exp=drum_diffusion trainer.gpus=1 datamodule.dataset.path=<path/to/your/train/data>
```

*NOTE:* To train on GPU, you must have a CUDA-capable GPU and have the CUDA toolkit installed for your specific to your system (ex. Linux, x86_64, WSL-Ubuntu). More information can be found [here](https://developer.nvidia.com/cuda-toolkit).


**Resume run from a checkpoint (with GPU):**

```bash
python train.py exp=drum_diffusion trainer.gpus=1 +ckpt=</path/to/checkpoint.ckpt> datamodule.dataset.path=<path/to/your/train/data>
```

## Repository Structure
The structure of this repository is as follows:
```
├── main
│   ├── diffusion_module.py     - contains pl model, data loading, and logging functionalities for training
│   └── utils.py                - contains utility functions for training
├── exp
│   └── *.yaml                  - Hydra configuration files
├── setup
│   ├── environment.yml         - file to set up conda environment
│   └── requirements.txt        - contains repo dependencies
├── images                      - directory containing images for README.md
│   └── *.png
├── samples                     - directory containing sample outputs from tiny-audio-diffusion models
│   └── *.wav
├── .env.tmp                    - temporary environment variables (rename to .env)
├── .gitignore
├── README.md
├── Inference.ipynb             - Jupyter notebook for running inference to generate new samples
├── config.yaml                 - Hydra base configs
├── train.py                    - script for training
├── data                        - directory to host custom training data
│   └── wav_dataset
│       └── (*.wav)
└── saved_models                - directory to host model checkpoints and hyper-parameters for inference
    └── (kicks/snare/etc.)
        ├── (*.ckpt)            - pl model checkpoint file
        └── (config.yaml)       - pl model hydra hyperparameters (required for inference)
```
