from typing import List, Tuple, Any, Dict
import re


def extract_defendants(sentences: List, tags: List, index: int, text: List[str]) -> Tuple[List, int]:
    defendants = []

    if not re.search('^[1-9].', text[index]):
        defendant = ''.join([i for i in text[index] if not i.isdigit() and i != '.'])
        defendants.append(defendant)
        sentences.append([defendant])
        tags.append(['Defendant'])
        return defendants, index + 1

    while re.search('^[1-9].', text[index]):
        defendant = ''.join([i for i in text[index] if not i.isdigit() and i != '.'])
        defendants.append(defendant)
        sentences.append([defendant])
        tags.append(['Defendant'])
        index += 1

    return defendants, index + 1


def extract_prosecutors(sentences: List, tags: List, index: int, text: List[str]) -> Tuple[List, int]:
    prosecutors = []

    while text[index] != 'נ' and text[index] != 'נ ג ד':
        prosecutor = ''.join([i for i in text[index] if not i.isdigit() and i != '.'])
        prosecutors.append(prosecutor)
        sentences.append([prosecutor])
        tags.append(['Prosecutor'])
        index += 1

    sentences.append(text[index])
    tags.append(['O'])

    return prosecutors, index+4


def extract_judges(sentences: List, tags: List, text: List[str]) -> Tuple[List, int]:
    judges, flag = [], False

    for i, raw_text in enumerate(text):
        if flag:
            if re.search(':$', raw_text):
                sentences.append(raw_text.split(" "))
                tags.append(['O'] * len(raw_text.split(" ")))
                return judges, i+1

        if re.search("^כבוד", raw_text):
            flag = True
            sentences.append([raw_text])
            judges.append(raw_text)
            tags.append(['Judge'])
        else:
            sentences.append(raw_text.split(" "))
            tags.append(['O'] * len(raw_text.split(" ")))


def extract_date(sentences: List, tags: List, index: int, text: List[str]) -> Any:

    while  ('ניתן היום' not in text[index]) and \
            not re.search("\u200f[0-9]+[.][0-9]+[.][0-9][0-9]+", text[index]) and \
            ('תאריך הישיבה:' not in text[index]):
        sentences.append(text[index].split(" "))
        tags.append(['O'] * len(text[index].split(" ")))
        index += 1
        if index == len(text):
            return None

    if re.search("\u200f[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",text[index]):
        s = re.search("\u200f[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",text[index])

        if s:
            sentences.append([s])
            tags.append(['Date'])
            return s[0].replace('\u200f', '')
        else:
            return None

    elif 'ניתן היום' in text[index] or 'תאריך הישיבה:' in text[index]:
        while not re.search("[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",text[index]):
            sentences.append(text[index].split(" "))
            tags.append(['O'] * len(text[index].split(" ")))
            index += 1
            if index == len(text):
                return None

        sentences.append([re.search("[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",text[index])[0]])
        tags.append(['Date'])
        return re.search("[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",text[index])[0]

    else:
        return None


def extract_ner(docs: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
    names, judges, prosecutors, defendants, dates, sentences, tags = [], [], [], [], [], [], []

    for name, doc in docs.items():
        names.append(name)

        judges_tup = extract_judges(sentences, tags, doc)
        judges.append(judges_tup[0])

        prosecutors_tup = extract_prosecutors(sentences, tags, judges_tup[1], doc)
        prosecutors.append(prosecutors_tup[0])

        defendants_tup = extract_defendants(sentences, tags, prosecutors_tup[1], doc)
        defendants.append(defendants_tup[0])

        date = extract_date(sentences, tags, defendants_tup[1], doc)
        dates.append(date)

    return {
        'Document' : names,
        'Judges' : judges,
        'Prosecutors' : prosecutors,
        'Defendants' : defendants,
        'Dates' : dates
    }, {
        'Sentences' : sentences,
        'Tags' : tags
    }


