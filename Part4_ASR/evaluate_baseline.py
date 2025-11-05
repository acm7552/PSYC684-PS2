# Harry Hennessy, Jack Mclaughlin, Sophia Caruana, Andrew Murphy
# PSYC 684
#
# This file loads the OpenAI Whisper Tiny model and the eka-medical-asr-evaluation-dataset
# Once loaded, the model is tested and compared to the finetuned WER obtained during the fine tuning process. Our model should perform better on ASR tasks involving medical vocabulary
# Currently configured for training using Torch's Direct ML for AMD GPUs.
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
from huggingface_hub import login
import os

# Configuration

# Initialize the DirectML device object
dml = torch_directml.device()
device = dml

# Define the global WER metric object
wer_metric = evaluate.load("wer")

# WER from fine-tuning process
FINETUNED_WER = 0.3622760385902737 

BASE_MODEL_ID = "openai/whisper-tiny"
OUTPUT_DIR = "./whisper-tiny-baseline-eval" 

hf_token = 'INSERT_HF_TOKEN' 


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


# Compute metrics function

def compute_metrics(pred):
    # Assumes preprocess_logits_for_metrics returns a PyTorch tensor, converting it to a NumPy array.
    pred_ids = pred.predictions[0] 
    label_ids = pred.label_ids

    # Filter out unexpected negative values
    pred_ids[pred_ids < 0] = 0
        
    # Explicitly cast to 32-bit integer type to satisfy tokenizer backend.
    pred_ids = pred_ids.astype(np.int32)
        
    # replace -100 with the pad token ID
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Preprocess logits

# Returns predicted token IDs as a PyTorch tensor, allowing Trainer's internal utilities to handle final conversion to numpy.
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
        
    # Apply argmax to get the predicted token IDs
    pred_ids = torch.argmax(logits, dim=-1)
    
    # Move the tensor to CPU memory if it's on a device (DML/CUDA/GPU). Trainer will then handle the final conversion to NumPy inside its loop.
    pred_ids = pred_ids.cpu()
    
    # Returns a tuple of predicted IDs, original labels
    return pred_ids, labels


if __name__ == '__main__':
    # Authenticate with Hugging Face
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face Hub.")

    # Load the dataset
    raw_dataset = load_dataset("ekacare/eka-medical-asr-evaluation-dataset", 'en')

    # Split the single test split into new train, validation, and test splits.
    train_val_split = raw_dataset["test"].train_test_split(
        test_size=0.2,
        seed=42
    )

    # Further split the test into validation and final test splits
    val_test_split = train_val_split["test"].train_test_split(
        test_size=0.5,
        seed=42
    )

    # Create the final DatasetDict, only need the validation split for evaluation
    dataset = DatasetDict({
        "validation": val_test_split["train"],
    })

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
        remove_columns=dataset.column_names["validation"],
        num_proc=4,
        fn_kwargs={"processor": processor}
    )

    print(f"Loading and Evaluating Baseline Model: {BASE_MODEL_ID}")

    # Load model onto the DML device
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID).to(device)

    # Define Training Arguments for evaluation only
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=8,
        load_best_model_at_end=False,
        push_to_hub=False
    )

    # Initialize Trainer and run evaluation
    data_collator = CustomDataCollator(processor=processor)

    trainer = Seq2SeqTrainer(
        model=base_model,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=dataset["validation"], 
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
    )

    # Run evaluation
    metrics = trainer.evaluate()

    # Print Final Result
    print("\n" + "=" * 60)
    print("ASR Model Performance Comparison")
    print("=" * 60)
    print(f"1. Fine-Tuned Whisper Large-v3 WER: {FINETUNED_WER:.4f}")
    print(f"2. Baseline {BASE_MODEL_ID} WER: {metrics['eval_wer']:.4f}")
    print("-" * 60)