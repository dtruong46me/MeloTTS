"""BERT feature extraction for Spanish text.

This module provides functionality to extract phone-level representations
using a pre-trained Spanish BERT model.
"""

import sys
from typing import Any, List, Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model: Optional[Any] = None


def get_bert_feature(text: str, word2ph: List[int], device: Optional[str] = None) -> Any:
    """Extracts BERT phone-level features for a given text.

    Args:
        text (str): The input Spanish text.
        word2ph (List[int]): A list representing the word-to-phoneme alignment counts.
        device (Optional[str], optional): The torch device to use. If None, it defaults to
            'mps' on Apple Silicon, or 'cuda' otherwise. Defaults to None.

    Returns:
        Any: A PyTorch tensor of transposed phone-level features.
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
        res_hidden = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res_hidden[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature_tensor = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature_tensor.T
