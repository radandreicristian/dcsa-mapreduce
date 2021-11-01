import re
import string
import nltk
from nltk.corpus import stopwords

languages = ['english', 'italian', 'french']
nltk.download('stopwords')
languages_stopwords = [stopwords.words(language) for language in languages]
all_stopwords = set([word for language_stopwords in languages_stopwords for word in language_stopwords])


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
    text_tokens = text.split()
    return " ".join([x for x in text_tokens if x not in all_stopwords])


def remove_numbers(text: str) -> str:
    """
    Remove the numbers from a string.

    :param text: A string that may contain numbers.
    :return: A string without numbers.
    """
    # From https://stackoverflow.com/a/68325680
    return text.translate(str.maketrans('', '', string.digits))


def remove_roman_numerals(text: str) -> str:
    """
    Remove the Roman numerals from a string.

    :param text: A text that may contain roman numerals.
    :return: A text without Roman numerals.
    """
    # From https://stackoverflow.com/a/68050802
    pattern = r"\b(?=[mdclxvii])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})([ii]x|[ii]v|v?[ii]{0,3})\b\.?"
    return re.sub(pattern, '', text, flags=re.I)


def preprocess_text(text: str) -> str:
    result = text.lower()
    methods = [remove_numbers, remove_roman_numerals, remove_stop_words, remove_punctuation]
    for method in methods:
        result = method(result)
    return result
