import string
import nltk
from nltk.corpus import stopwords
import regex as re


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from a string.
    :param text: A text string that may contain punctuation.
    :return: The text string without punctuation.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(text: str) -> str:
    """
    Remove the stop words from a string.

    :param text: A text string that may contain stop words, which we do not want for our ML model to process.
    :return: The text string without stop words.
    """
    language_stopwords = set(stopwords.words('english'))
    text_tokens = text.split()
    return " ".join([x for x in text_tokens if x not in language_stopwords])


def remove_years(text: str) -> str:
    return text.translate(str.maketrans('', '', string.digits))


def clear_text(text: str) -> str:
    text = text.lower()
    steps = [remove_years, remove_stop_words, remove_punctuation]
    for step in steps:
        text = step(text)
    return text
