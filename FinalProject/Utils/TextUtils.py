import string
from typing import List

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration


def list_of_words(text: str) -> List[str]:
    text = text.replace(u"ר ש מ ת", u"רשמת")
    text = text.replace(u"ש ו פ ט", u"שופט")
    text = text.replace(u"ש ו פ ט ת", u"שופטת")
    text = text.replace(u"ה נ ש י א ה", u"הנשיאה")
    text = text.replace(u"נ ג ד", u"נגד")

    punctuation = tuple(string.punctuation)
    raw_words = []
    words = text.split()
    hebrew_alphabet = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ"

    for i in range(len(words)):
        word = words[i]
        while word.startswith(punctuation):
            word = word[1:]

        while word.endswith(punctuation):
            word = word[:-1]

        if word != "" and any((c in word) for c in hebrew_alphabet):
            raw_words.append(word)

    return raw_words


def html_scrape(filepath: str) -> List[str]:
    with open(filepath, "rb") as f:
        rv = []
        soup = BeautifulSoup(f, 'html.parser')
        texts = soup.findAll(string=True)
        for text in texts:
            if tag_visible(text) and not (text.isspace() or text == ""):
                rv.append(text)

        return rv


def tag_visible(element) -> bool:
    types = ['style', 'script', 'head', 'title', 'meta', '[document]']
    if element.parent.name in types:
        return False
    if isinstance(element, Comment) or isinstance(element, Declaration):
        return False
    return True
