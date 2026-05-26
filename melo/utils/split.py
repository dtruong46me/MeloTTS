"""Text-splitting utilities for multi-language TTS preprocessing.

This module provides functions to split long text strings into shorter,
TTS-friendly chunks while attempting to keep sentences intact. It supports
Latin-script languages (EN, FR, ES, SP) and Chinese (ZH), with dedicated
helpers for merging overly short sentences.
"""

from __future__ import annotations

import re
from typing import List


def split_sentence(text: str, min_len: int = 10, language_str: str = 'EN') -> List[str]:
    """Route text splitting to the appropriate language-specific function.

    Args:
        text: Input text to split.
        min_len: Minimum desired length (in characters) for each sentence chunk.
        language_str: Language code. Use ``'EN'``, ``'FR'``, ``'ES'``, or
            ``'SP'`` for Latin-script languages; any other value routes to the
            Chinese splitter.

    Returns:
        List of sentence strings after splitting and merging short chunks.
    """
    if language_str in ['EN', 'FR', 'ES', 'SP']:
        sentences = split_sentences_latin(text, min_len=min_len)
    else:
        sentences = split_sentences_zh(text, min_len=min_len)
    return sentences


def split_sentences_latin(text: str, min_len: int = 10) -> List[str]:
    """Split Latin-script text into sentence chunks via :func:`txtsplit`.

    Normalises common CJK punctuation to ASCII equivalents, strips angle
    brackets and quotation marks, then delegates to :func:`txtsplit`.

    Args:
        text: Input text (English, French, Spanish, etc.).
        min_len: Minimum desired chunk length (passed through; not directly
            used in this function but kept for signature parity).

    Returns:
        Non-empty, stripped sentence strings.
    """
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[""]', '\"', text)
    text = re.sub('['']', "'", text)
    text = re.sub(r"[<>\(\)\[\]\"«»]+", "", text)
    return [item.strip() for item in txtsplit(text, 256, 512) if item.strip()]


def split_sentences_zh(text: str, min_len: int = 10) -> List[str]:
    """Split Chinese text into sentence chunks of at least ``min_len`` characters.

    Normalises punctuation, inserts split markers after punctuation, then
    accumulates sentences until each chunk exceeds ``min_len`` characters
    before delegating final merging to :func:`merge_short_sentences_zh`.

    Args:
        text: Input Chinese (or mixed) text.
        min_len: Minimum number of characters a chunk must reach before it is
            committed as a separate sentence.

    Returns:
        List of merged sentence strings.
    """
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    # 将文本中的换行符、空格和制表符替换为空格
    text = re.sub('[\n\t ]+', ' ', text)
    # 在标点符号后添加一个空格
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    # 分隔句子并去除前后空格
    # sentences = [s.strip() for s in re.split('(。|！|？|；)', text)]
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def merge_short_sentences_en(sens: List[str]) -> List[str]:
    """Avoid short sentences by merging them with the following sentence.

    Args:
        sens: List of input sentences.

    Returns:
        List of output sentences with short entries merged into their
        neighbour.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out


def merge_short_sentences_zh(sens: List[str]) -> List[str]:
    # return sens
    """Avoid short sentences by merging them with the following sentence.

    Args:
        sens: List of input sentences.

    Returns:
        List of output sentences with short entries (≤ 2 characters) merged
        into their neighbour.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out


def txtsplit(text: str, desired_length: int = 100, max_length: int = 200) -> List[str]:
    """Split text into chunks of a desired length trying to keep sentences intact.

    Args:
        text: Input text to split.
        desired_length: Target chunk length in characters. The splitter tries
            to commit a chunk once it reaches this length at a natural boundary.
        max_length: Hard upper limit on chunk length. If no natural boundary
            is found the splitter will force a split here.

    Returns:
        List of non-empty text chunks that do not consist solely of
        whitespace or punctuation.
    """
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'([,.?!])', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta: int) -> str:
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta: int) -> str:
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit() -> None:
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                d = pos - split_pos[-1]
                seek(-d)
            else:
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        elif not in_quote and (c in '!?\n' or (c in '.,' and peek(1) in '\n ')):
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]
    return rv


if __name__ == '__main__':
    zh_text = "好的，我来给你讲一个故事吧。从前有一个小姑娘，她叫做小红。小红非常喜欢在森林里玩耍，她经常会和她的小伙伴们一起去探险。有一天，小红和她的小伙伴们走到了森林深处，突然遇到了一只凶猛的野兽。小红的小伙伴们都吓得不敢动弹，但是小红并没有被吓倒，她勇敢地走向野兽，用她的智慧和勇气成功地制服了野兽，保护了她的小伙伴们。从那以后，小红变得更加勇敢和自信，成为了她小伙伴们心中的英雄。"
    en_text = """I didn't know what to do. I said please kill her because it would be better than being kidnapped," Ben, whose surname CNN is not using for security concerns, said on Wednesday. "It's a nightmare. I said 'please kill her, don't take her there.'"""
    sp_text = "¡Claro! ¿En qué tema te gustaría que te hable en español? Puedo proporcionarte información o conversar contigo sobre una amplia variedad de temas, desde cultura y comida hasta viajes y tecnología. ¿Tienes alguna preferencia en particular?"
    fr_text = "Bien sûr ! En quelle matière voudriez-vous que je vous parle en français ? Je peux vous fournir des informations ou discuter avec vous sur une grande variété de sujets, que ce soit la culture, la nourriture, les voyages ou la technologie. Avez-vous une préférence particulière ?"

    print(split_sentence(zh_text, language_str='ZH'))
    print(split_sentence(en_text, language_str='EN'))
    print(split_sentence(sp_text, language_str='SP'))
    print(split_sentence(fr_text, language_str='FR'))
