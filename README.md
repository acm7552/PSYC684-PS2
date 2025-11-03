# PSYC684-PS2
PSYC684 Problem Set 2 Repository

***

## Part 2

Initialize python environment and run depending on if you want to compare all text files:
  
`python DTW_algorithm.py --all`  

OR  
`python DTW_algorithm.py --mystery`  
  
to compare mystery files to templates only.
***
## Part 3:

Run in **"\Part3_Grapheme_to_phoneme\\"** directory:  
`python lett2phon.py`

## Part 4

It is highly recommended to create a new Conda environment when fine-tuning. To do so, install Conda and run the following:

`conda create -n env-name python=3.10`

`conda activate env-name`

To install the necessary libraries:

`conda install -c conda-forge pytorch torchaudio pyarrow ffmpeg -y`

`pip install transformers datasets accelerate peft bitsandbytes evaluate`

`pip install torch-directml librosa soundfile`

`pip install -U transformers peft datasets evaluate huggingface_hub`

Once the proper dependencies have been installed, to train a model, run:

`python finetune_asr.py`
