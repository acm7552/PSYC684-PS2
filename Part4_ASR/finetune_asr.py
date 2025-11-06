# Harry Hennessy, Jack Mclaughlin, Sophia Caruana, Andrew Murphy
# PSYC 684
#
# This file loads the OpenAI Whisper Tiny model and the eka-medical-asr-evaluation-dataset,
# fine-tuning the Whisper model on medical data.
# Once pre-trained, the model should perform better on ASR tasks involving medical vocabulary
# Currently configured for training using Torch's Direct ML for AMD GPUs. Instructions are left for converting for NVIDIA usage -H

from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login
import torch
import torch_directml
import evaluate
import numpy as np
import re
import os
import sys
from packaging import version

# Should prevent crashing!
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

# For AMD: Initialize the DirectML device object
# For NVIDIA: Replace with CUDA
dml = torch_directml.device()

# Define the token and model ID. Replace 'INSERT_HF_TOKEN' with your token after creating HF account and gaining access to eka-medical-asr-evaluation-dataset
hf_token = 'INSERT_HF_TOKEN'
model_id = "openai/whisper-tiny"
wer_metric = evaluate.load("wer")

ADAPTER_DIR = "./whisper-medical-finetuned-adapter"
MERGED_DIR  = "./whisper-medical-merged-model"
OUTPUT_DIR  = "./whisper-medical-finetuned"
RESUME_CKPT = os.path.join(OUTPUT_DIR, "checkpoint-3620") #replace with wherever you want to resume training, or "" if beginning from scratch

# Custom Data Collator for ASR Data. Based on transformers DataCollatorSpeechSeq2SeqWithPadding
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Pad input features (spectrograms)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", return_attention_mask=True)

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

# Returns predicted token IDs as a PyTorch tensor, allowing Trainer's internal utilities to handle final conversion to numpy.
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1).cpu()
    return pred_ids, labels

# Normalize text before WER to avoid punctuation/case noise
def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("’", "'").replace("`", "'").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Remove optimizer/scheduler state files from a checkpoint so Trainer won’t torch.load() them. (For DML compatibility)
def _strip_optimizer_files(ckpt_dir: str):
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return
    for fname in ("optimizer.pt", "optimizer.bin", "optimizer_state.pt", "scheduler.pt", "scheduler.bin"):
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"Removed: {fpath}")
            except Exception as e:
                print(f"Could not remove {fpath}: {e}")

# Prepare dataset for training
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

# Set evaluation metric to Word Error Rate
def build_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        label_ids = pred.label_ids
        pred_ids[pred_ids < 0] = 0
        pred_ids = pred_ids.astype(np.int32)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [_norm_text(s) for s in pred_str]
        label_str = [_norm_text(s) for s in label_str]
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    return compute_metrics

# Main block
if __name__ == '__main__':

    # Authenticate with Hugging Face
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face Hub.")

    # Load the dataset
    raw_dataset = load_dataset("ekacare/eka-medical-asr-evaluation-dataset", 'en')

    # Split single test into train/val/test
    train_val_split = raw_dataset["test"].train_test_split(test_size=0.2, seed=42)
    val_test_split  = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        "train": train_val_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_id)

    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    base_model.to(dml)

    # Apply LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # Print trainable parameter summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}\n All parameters: {total_params}\n Trainable %: {100 * trainable_params / total_params:.2f}")

    # Preprocess all splits
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=4,
        fn_kwargs={"processor": processor}
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_train_epochs=10,
        logging_steps=50,
        eval_strategy="steps",         
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        do_train=True,
        do_eval=True,
        fp16=False,                     # NVIDIA can set True
        load_best_model_at_end=True,    
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        save_safetensors=True,         
    )

    data_collator   = CustomDataCollator(processor)
    compute_metrics = build_compute_metrics(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
    )

    # Directml workaround
    # If torch < 2.6, Trainer will refuse to torch.load() optimizer/scheduler due to CVE-2025-32434.
    # Remove optimizer/scheduler files from the checkpoint so resume only loads model weights.
    resume_dir = RESUME_CKPT if os.path.isdir(RESUME_CKPT) else None
    if resume_dir:
        torch_ver = version.parse(torch.__version__.split("+")[0])
        if torch_ver < version.parse("2.6"):
            print(f"Resuming from {resume_dir} on torch {torch.__version__} — stripping optimizer/scheduler to avoid torch.load()")
            _strip_optimizer_files(resume_dir)
        trainer.train(resume_from_checkpoint=resume_dir)
    else:
        trainer.train()

    # Save best model
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    print("Saved adapter to:", ADAPTER_DIR)

    # Export merged model
    fresh_base = WhisperForConditionalGeneration.from_pretrained(model_id)
    lora_loaded = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)
    merged_model = lora_loaded.merge_and_unload().to("cpu")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR)
    processor.save_pretrained(MERGED_DIR)
    print("Merged model saved in:", MERGED_DIR)

    print("Fine-tuning complete.")
