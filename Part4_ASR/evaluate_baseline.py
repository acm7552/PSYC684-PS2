# Harry Hennessy, Jack Mclaughlin, Sophia Caruana, Andrew Murphy
# PSYC 684
#
# Evaluates baseline Whisper Tiny and a fine-tuned adapter on the eka-medical-asr-evaluation-dataset
# Using the exact same evaluation style as the old single-eval script.

import torch
import numpy as np
import evaluate
import torch_directml
from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel
from huggingface_hub import login
import os
import re

# Configuration
BASE_MODEL_ID = "openai/whisper-tiny"
MERGED_DIR = "./whisper-medical-merged-model"
FINETUNED_ADAPTER_PATH = "./whisper-medical-finetuned-adapter"
OUTPUT_DIR = "./whisper-tiny-eval"
hf_token = 'INSERT_HF_TOKEN'

# Initialize the DirectML device object
device = torch_directml.device()

# Define the global WER metric object
wer_metric = evaluate.load("wer")


# Data collator from finetune_asr.py
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Pad input features (spectrograms)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels (tokens)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Prevent loss being calculated on the starting token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# -Normalize text for comparison
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Compute metrics function
def compute_metrics(pred):
    # Assumes preprocess_logits_for_metrics returns a PyTorch tensor, converting it to a NumPy array.
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.cpu().numpy()
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.cpu().numpy()

    # Filter out unexpected negative values
    pred_ids[pred_ids < 0] = 0

    # Explicitly cast to 32-bit integer type to satisfy tokenizer backend.
    pred_ids = pred_ids.astype(np.int32)

    # replace -100 with the pad token ID
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    norm_preds = [normalize_text(t) for t in pred_str]
    norm_refs  = [normalize_text(t) for t in label_str]

    wer = wer_metric.compute(predictions=norm_preds, references=norm_refs)
    return {"wer": wer}

# Returns predicted token IDs as a PyTorch tensor, allowing Trainer's internal utilities to handle final conversion to numpy.
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # Apply argmax to get the predicted token IDs
    pred_ids = torch.argmax(logits, dim=-1)

    # Returns a tuple of predicted IDs, original labels
    return pred_ids, labels

if __name__ == "__main__":
    # Authenticate with Hugging Face
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face Hub.")

    # Load the dataset
    raw_dataset = load_dataset("ekacare/eka-medical-asr-evaluation-dataset", 'en')

    # Split the single test split into new train, validation, and test splits.
    train_val_split = raw_dataset["test"].train_test_split(test_size=0.2, seed=42)

    # Further split the test into validation and final test splits
    val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

    # Create the final DatasetDict, only need the validation split for evaluation
    dataset = DatasetDict({"validation": val_test_split["train"]})

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    # Prepare dataset for training (using the same preprocessing function from finetune_asr.py)
    def prepare_dataset(batch, processor):
        # Load audio
        audio = batch["audio"]

        # Compute log-Mel spectrogram and input features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode transcripts
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Apply preprocessing to validation split
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["validation"].column_names,
        num_proc=4,
        fn_kwargs={"processor": processor}
    )

    data_collator = CustomDataCollator(processor=processor)

    # Baseline
    print(f"Evaluating baseline model: {BASE_MODEL_ID}")

    # Load model onto the DML device
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID).to(device)

    # Define Training Arguments for evaluation only
    base_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "baseline"),
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=8,
        load_best_model_at_end=False,
        push_to_hub=False
    )

    # Initialize Trainer and run evaluation
    trainer_base = Seq2SeqTrainer(
        model=base_model,
        args=base_args,
        compute_metrics=compute_metrics,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Run evaluation
    base_metrics = trainer_base.evaluate()
    print(f"\nBaseline WER: {base_metrics['eval_wer']:.4f}")

    # Fine-tuned
    print("\nEvaluating fine-tuned adapter ...")

    # Load model onto the DML device
    finetuned = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR).to(device)

    # Define Training Arguments for evaluation only
    ft_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "finetuned"),
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=8,
        load_best_model_at_end=False,
        push_to_hub=False
    )

    # Initialize Trainer and run evaluation
    trainer_ft = Seq2SeqTrainer(
        model=finetuned,
        args=ft_args,
        compute_metrics=compute_metrics,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Run evaluation
    ft_metrics = trainer_ft.evaluate()
    print(f"\nFine-Tuned Adapter WER: {ft_metrics['eval_wer']:.4f}")

    # Print Final Result
    print("\nASR Model Performance Comparison")
    print(f"Baseline WER: {base_metrics['eval_wer']:.4f}")
    print(f"Fine-Tuned Adapter WER: {ft_metrics['eval_wer']:.4f}")

