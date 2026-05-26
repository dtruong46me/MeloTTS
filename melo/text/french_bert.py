"""Module for extracting French BERT features for MeloTTS."""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys
from typing import List, Optional, Any

model_id = "dbmdz/bert-base-french-europeana-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = None


def get_bert_feature(text: str, word2ph: List[int], device: Optional[str] = None) -> torch.Tensor:
    """Extracts BERT features for the provided French text.

    Args:
        text (str): The input text to process.
        word2ph (List[int]): The mapping of words to phonemes.
        device (Optional[str], optional): The compute device to use (e.g., "cuda", "cpu"). Defaults to None.

    Returns:
        torch.Tensor: The phone-level BERT features extracted from the model.
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
    phone_level_feature: List[torch.Tensor] = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature_tensor = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature_tensor.T
