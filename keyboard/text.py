"""Classical text processing (area B): regex normalization and word tokenization.

Every model sees text that went through `normalize`, so the four models are
trained and tested on exactly the same characters.
"""
from __future__ import annotations

import re

# WikiText writes "1 @,@ 000" for "1,000", "well @-@ known" for "well-known" ...
_WIKITEXT_FIXES = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")]

# A "word" for counting and pre-tokenizing: a run of letters/digits, or one
# punctuation character.  `\w` is Unicode-aware, so Turkish letters work too.
WORD_RE = re.compile(r"\w+|[^\w\s]")

# Letters only (no digits, no underscore): what the keyboard may suggest.
LETTERS_RE = re.compile(r"^[^\W\d_]+$")

# The word the user is still typing at the end of the text box.
_PARTIAL_RE = re.compile(r"([^\W\d_]+)$")


def detokenize_wikitext(line: str) -> str:
    """Undo WikiText's spacing so the text looks like normal typing."""
    for old, new in _WIKITEXT_FIXES:
        line = line.replace(old, new)
    line = re.sub(r" ([.,;:!?%)\]}])", r"\1", line)        # "word ," -> "word,"
    line = re.sub(r"([(\[{$]) ", r"\1", line)               # "( word" -> "(word"
    line = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", line)  # "do n't" -> "don't"
    line = re.sub(r'" ([^"]*?) "', r'"\1"', line)           # '" quoted "' -> '"quoted"'
    return line


def lowercase(text: str, language: str = "en") -> str:
    """Lower-case with the Turkish dotted/dotless i rule when needed."""
    if language == "tr":
        text = text.replace("I", "ı").replace("İ", "i")
    return text.lower()


def normalize(text: str, language: str = "en") -> str:
    """Lower-case and collapse all whitespace to single spaces."""
    text = lowercase(text, language)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    """Regex word tokenizer: words and single punctuation marks."""
    return WORD_RE.findall(text)


def is_word(token: str) -> bool:
    """True for a token made only of letters (a valid suggestion)."""
    return bool(LETTERS_RE.match(token))


def split_partial(text: str) -> tuple[str, str]:
    """Split the text box into (finished context, word being typed).

    "the quick bro" -> ("the quick ", "bro")
    "the quick "    -> ("the quick ", "")
    """
    match = _PARTIAL_RE.search(text)
    if not match:
        return text, ""
    start = match.start(1)
    if start > 0 and not text[start - 1].isspace():
        return text, ""          # e.g. "don't": the letters after ' are not a new word
    return text[:start], match.group(1)
