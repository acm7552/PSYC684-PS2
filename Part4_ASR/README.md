It is highly recommended to create a new Conda environment when fine-tuning. To do so, run the following:

`conda create -n env-name python=3.10`

`conda activate env-name`

To install the necessary libraries:

`conda install -c conda-forge pytorch torchaudio pyarrow ffmpeg -y`

`pip install transformers datasets accelerate peft bitsandbytes evaluate`

`pip install torch-directml librosa soundfile`

`pip install -U transformers peft datasets evaluate huggingface_hub`

Once the proper dependencies have been installed, to train a model, run:

`python finetune_asr.py`
