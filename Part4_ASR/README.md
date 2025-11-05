It is highly recommended to create a new Conda environment based on the environment.yml file providedwhen fine-tuning. To do so, run the following:

`conda env create -f environment.yml`

`conda activate whisper-ft-new`

Once the environment has been installed, to train a model, run:

`python finetune_asr.py`

In order to compare the performance of the fine-tuned model with the baseline, run:

`python evaluate_baseline.py`

NOTE: WER for fine-tuned model is manually inserted for this file. If a new model has been fine-tuned, please change the FINETUNED_WER variable in the code.
