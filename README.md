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

It is highly recommended to create a new Conda environment based on the environment.yml file provided when fine-tuning. To do so, download environment.yml, and from its download folder, run the following:

`conda env create -f environment.yml`

`conda activate whisper-ft-new`

Once the environment has been installed, to train a model, create an account on HuggingFace and request access to the eka-medical-asr-evaluation dataset. You will also need to create a token. Copy this token into `INTERT_HF_TOKEN`, then run:

`python finetune_asr.py`

In order to compare the performance of the fine-tuned model with the baseline, run:

`python evaluate_baseline.py`

Audio Sample Transcription can be done using inference.py with the audio file as a command argument, as seen below:

`python inference.py <audio-file>`

This can be done in batches with:

`batch_inference_compare.py`

Examining audio files in the 'inferences' subdirectory

To compare to models to a standard out-of-domain test set, use:

`standard_comparison.py`
