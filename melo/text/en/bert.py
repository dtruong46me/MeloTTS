"""Module for extracting BERT features from English text.

This module uses a pre-trained BERT model to extract phone-level features
for text-to-speech synthesis.
"""

import sys
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = None


def get_bert_feature(
    text: str, word2ph: List[int], device: Optional[str] = None
) -> torch.Tensor:
    """Extract BERT features for the given text based on word-to-phone mapping.

    Args:
        text (str): The input English text.
        word2ph (List[int]): A list indicating the number of phones each word maps to.
        device (Optional[str], optional): The device to run the model on. Defaults to None.

    Returns:
        torch.Tensor: The phone-level BERT feature tensor.
    """
    global model
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature_tensor = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature_tensor.T
