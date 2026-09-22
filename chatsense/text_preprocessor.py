import unicodedata
import re
import emoji


_WHITESPACE_RE = re.compile(r'\s+')
# Repeated chars: collapse runs of 3+ to 2 (sooo -> soo, keeps the emphasis)
_REPEATED_CHARS_RE = re.compile(r'(.)\1{2,}')


class TextPreprocessor:
    """
    Cleans and normalises anonymised message text before sentiment analysis.

    Rules applied (in order):
    1. Unicode NFKC normalisation
    2. Whitespace normalisation (collapse to single space)
    3. Emoji -> :alias: conversion so VADER can interpret them
    4. Repeated character reduction (sooo -> soo) — keeps emphasis signal
    5. Strip leading/trailing whitespace

    NOT applied (intentionally):
    - Stopword removal (would destroy negation signals like "not", "never")
    - Lowercasing (VADER uses capitalisation as emphasis signal)
    - Punctuation removal (!!!  and ... carry sentiment weight)
    - Lemmatisation (would alter "not" -> "not", but risks edge cases)
    """

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ''

        # 1. Unicode normalisation
        text = unicodedata.normalize('NFKC', text)

        # 2. Whitespace normalisation
        text = _WHITESPACE_RE.sub(' ', text)

        # 3. Emoji to text alias (e.g. 😂 -> :face_with_tears_of_joy:)
        #    VADER's lexicon includes many emoji aliases
        text = emoji.demojize(text, delimiters=(':', ':'))

        # 4. Collapse excessive repeated characters (lol!!!!! -> lol!!)
        #    Keeps up to 2 repetitions so emphasis is not fully lost
        text = _REPEATED_CHARS_RE.sub(r'\1\1', text)

        return text.strip()
