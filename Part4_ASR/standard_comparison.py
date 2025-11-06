# Harry Hennessy, Jack Mclaughlin, Sophia Caruana, Andrew Murphy
# PSYC 684
#
# This file loads the OpenAI Whisper Tiny model, the fine tuned model, and librispeech's test set
# Once loaded, both models are tested on this standard test set to compare performance out of domain
import os
import re
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio, disable_caching, load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Configuration
device = torch.device("cpu")
BASE_MODEL_ID = "openai/whisper-tiny"
FINETUNED_ADAPTER_PATH = "./whisper-medical-finetuned-adapter"
MERGED_DIR = "./whisper-medical-merged-model"
CACHE_DIR = "./librispeech_cached"
NUM_PROC = 4
BATCH_SIZE = 8
DEBUG_PRINT = 8

wer_metric = evaluate.load("wer")

# normalization used for LibriSpeech-style WER
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    # unify whitespace and quotes
    text = text.replace("’", "'").replace("`", "'")
    text = text.lower()
    # remove punctuation except keep simple apostrophes inside words
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Suppress tokens for timestamps if needed
def get_suppress_tokens(model, processor):
    suppress = []
    if hasattr(model.config, "suppress_tokens") and model.config.suppress_tokens:
        suppress = list(model.config.suppress_tokens)
    try:
        suppress += list(processor.tokenizer._get_all_timestamp_tokens_id())
    except Exception:
        pass
    try:
        no_ts = processor.tokenizer.no_timestamps_token_id
        if no_ts not in suppress:
            suppress.append(no_ts)
    except Exception:
        pass
    return list({int(x) for x in suppress if x is not None})

# compute input features and store tokenized labels.
def prepare_dataset(example, processor):
    audio = example["audio"]
    example["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # store the original text
    example["text"] = example.get("text", "")
    # store token ids for labels
    example["labels"] = processor.tokenizer(example["text"]).input_ids
    return example

# Collation
def collate_fn(batch, processor, device):
    input_feature_list = [row["input_features"] for row in batch]
    text_labels = [row.get("text") or "" for row in batch]

    padded = processor.feature_extractor.pad(
        [{"input_features": f} for f in input_feature_list],
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = padded["input_features"].to(device)
    attention_mask = padded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    tokenized = processor.tokenizer(text_labels, padding=True, return_tensors="pt")
    labels = tokenized.input_ids
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    labels = labels.clone()
    labels[labels == pad_id] = -100
    labels = labels.to(device)

    return {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "labels": labels,
        "raw_texts": text_labels,
    }

# Get tokens from labels
def decode_labels(labels_tensor, processor):
    labels = labels_tensor.clone().cpu()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    labels[labels == -100] = pad_id
    return processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

# Evaluation
def evaluate_model_iter(model, processor, dataset, model_name, device, max_eval_examples=None):
    model.eval()
    preds, refs = [], []
    suppress_tokens = get_suppress_tokens(model, processor)
    print(f"\nEvaluating {model_name} ... Suppressing {len(suppress_tokens)} tokens.")
    bs = BATCH_SIZE
    total = len(dataset) if max_eval_examples is None else min(len(dataset), max_eval_examples)

    pbar = tqdm(dataset.iter(batch_size=bs), total=(total + bs - 1) // bs, desc=f"Inference ({model_name})")
    seen = 0
    printed = 0

    for batch_dict in pbar:
        rows = [dict(zip(batch_dict.keys(), v)) for v in zip(*batch_dict.values())]
        if seen + len(rows) > total:
            rows = rows[: (total - seen)]

        c = collate_fn(rows, processor, device)

        if seen == 0:
            print("DEBUG shapes: input_features", c["input_features"].shape,
                  "attention_mask", None if c["attention_mask"] is None else c["attention_mask"].shape,
                  "labels", c["labels"].shape)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=c["input_features"],
                attention_mask=c["attention_mask"],
                max_new_tokens=256,
                language="english",
                task="transcribe",
                suppress_tokens=suppress_tokens,
            )

        decoded_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_refs = decode_labels(c["labels"], processor)

        # Debug: print both raw and normalized pairs for the first few examples for inspection
        # for raw_p, raw_r in zip(decoded_preds, decoded_refs):
        #     if printed < DEBUG_PRINT:
        #         norm_p = normalize_text(raw_p)
        #         norm_r = normalize_text(raw_r)
        #         print(f"RAW PRED: {raw_p}")
        #         print(f"RAW REF : {raw_r}")
        #         print(f"NORM PRED: {norm_p}")
        #         print(f"NORM REF : {norm_r}")
        #         print("---")
        #         printed += 1
        #     else:
        #         break

        preds.extend(decoded_preds)
        refs.extend(decoded_refs)

        seen += len(rows)
        if seen >= total:
            break

    # Normalize before metric computation
    norm_preds = [normalize_text(t) for t in preds]
    norm_refs = [normalize_text(t) for t in refs]

    wer_score = wer_metric.compute(predictions=norm_preds, references=norm_refs)
    print(f"{model_name} WER: {wer_score*100:.3f}%")
    return wer_score

def main():
    print(f"Evaluating on LibriSpeech Test Clean")
    disable_caching()

    # load processor
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # load or preprocess dataset
    if os.path.exists(CACHE_DIR):
        print("Loading preprocessed dataset from disk cache:", CACHE_DIR)
        ds = load_from_disk(CACHE_DIR)
    else:
        ds = load_dataset(
            "librispeech_asr",
            "clean",
            split="test",
            verification_mode="no_checks"
        ).cast_column("audio", Audio(sampling_rate=16000))

        print("Preprocessing dataset...")
        ds = ds.map(
            lambda ex: prepare_dataset(ex, processor),
            remove_columns=[c for c in ds.column_names if c not in ("input_features", "labels", "text")],
            num_proc=NUM_PROC,
        )
        print("Saving preprocessed dataset to disk for future runs...")
        ds.save_to_disk(CACHE_DIR)

    # load models
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID).to(device)

    finetuned = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR).to(device)

    quick_check = False
    max_eval = 12 if quick_check else None

    wer_base = evaluate_model_iter(base, processor, ds, "Baseline", device, max_eval_examples=max_eval)
    wer_ft = evaluate_model_iter(finetuned, processor, ds, "Fine-tuned", device, max_eval_examples=max_eval)

    print("\nSummary:")
    print(f"Baseline: {wer_base*100:.2f}%")
    print(f"Fine-tuned: {wer_ft*100:.2f}%")

if __name__ == "__main__":
    main()

