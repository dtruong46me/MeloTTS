"""Module for extracting BERT features for Japanese text.

This module provides functions to process Japanese text using a pre-trained
BERT model and extract phoneme-level features for Text-To-Speech systems.
"""

import sys
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


models = {}
tokenizers = {}


def get_bert_feature(
    text: str,
    word2ph: list[int],
    device: Optional[str] = None,
    model_id: str = "tohoku-nlp/bert-base-japanese-v3",
) -> torch.Tensor:
    """Extracts BERT features for the given Japanese text at the phoneme level.

    Args:
        text (str): The input Japanese text.
        word2ph (list[int]): A list mapping words to phoneme counts.
        device (Optional[str], optional): The device to run the model on ('cuda', 'cpu', or 'mps'). Defaults to None.
        model_id (str, optional): The Hugging Face model ID to use. Defaults to 'tohoku-nlp/bert-base-japanese-v3'.

    Returns:
        torch.Tensor: The extracted BERT features at the phoneme level.
    """
    global model
    global tokenizer

    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model_id not in models:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(
            device
        )
        models[model_id] = model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizers[model_id] = tokenizer
    else:
        model = models[model_id]
        tokenizer = tokenizers[model_id]


    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        tokenized = tokenizer.tokenize(text)
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert inputs["input_ids"].shape[-1] == len(word2ph), f"{inputs['input_ids'].shape[-1]}/{len(word2ph)}"
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
