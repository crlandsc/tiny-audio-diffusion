# Tiny Audio Diffusion
This is a repository for generating and training short audio samples on less than 2GB VRAM.

The purpose of this project is to provide access to audio diffusion code for those interested in exploration that have limited resources. Because of the limited sample size, drum samples are a natural fit for training and generating.

The repository is built heavily adapting code from [Archinet's audio-diffusion-pytorch (v0.1.3)](https://github.com/archinetai/audio-diffusion-pytorch).

***NOTE:*** *THIS IS A WORK IN PROGRESS AND MAY BREAK. FUTURE UPDATES WILL IMPROVE CLEANLINESS AND ROBUSTNESS OF CODE.*

## Setup

#### Create virtual environment:

Run the following lines from the command line:
```bash
conda env create -f environment.yml
conda activate tiny-audio-diffusion
```

This will create a conda environment and install the ependincies in the utils/requirements.txt.


#### Install Python kernel for Jupyter Notebook
Run the following line to create a kernel for the current environment to run the inference notebook.

```bash
python -m ipykernel install --user --name tiny-audio-diffusion --display-name "tiny-audio-diffusion (Python 3.10)"
```


#### Define environment variables
Rename `.env.tmp` to `.env` and replace the entries with your own variables (example values are random)
```bash
DIR_LOGS=/logs
DIR_DATA=/data

# Required if using wandb logger
WANDB_PROJECT=audioproject
WANDB_ENTITY=johndoe
WANDB_API_KEY=a21dzbqlybbzccqla4txa21dzbqlybbzccqla4tx
```

## Inference
Open the `Inference.ipynb` in Jupyter Notebook to generate new drum samples.

## Train
Run the following commands in terminal to train the model.

`drum_diffusion.yaml` is contains the default model configuration. Additional custom model configurations can be added to the [`exp`](exp/) folder.

Train model from scratch:

```bash
python train.py exp=drum_diffusion datamodule.dataset.path=<path/to/your/train/data>
```

Run on GPU(s)

```bash
python train.py exp=drum_diffusion trainer.gpus=1 exp=drum_diffusion datamodule.dataset.path=<path/to/your/train/data>
```

Resume run from a checkpoint (with GPU):

```bash
python train.py exp=drum_diffusion trainer.gpus=1 +ckpt=</path/to/checkpoint.ckpt> exp=drum_diffusion datamodule.dataset.path=<path/to/your/train/data>
```