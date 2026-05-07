import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

MODEL_NAME = os.getenv("MODEL_NAME", "./roberta_local")
INPUT_FILE = Path(os.getenv("INPUT_FILE", "./data2022.xlsx"))
SAVE_PATH = Path(os.getenv("SAVE_PATH", "./processed_data2022.pt"))
MAX_LEN = 256


def run_preprocessing() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    df = pd.read_excel(INPUT_FILE)

    content = df["Qsubj"].fillna("").astype(str).tolist()
    reply = df["Reply"].fillna("").astype(str).tolist()
    labels = df["Isviolated"].fillna(0).astype(float).tolist()

    combined_texts = [f"{q} [SEP] {r}" for q, r in zip(content, reply)]
    encodings = tokenizer(
        combined_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    data = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float32),
    }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, SAVE_PATH)


if __name__ == "__main__":
    run_preprocessing()