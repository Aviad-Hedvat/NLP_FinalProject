from typing import Dict, List, Tuple
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import string

from TextUtils import list_of_words

from keras.utils import pad_sequences
from keras import layers as L
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from livelossplot.inputs.tf_keras import PlotLossesCallback


pos_words = [
    "מתקבל", "מתקבלת", "מתקבלים", "מתקבלות", "מקבל",
    "מקבלת", "מקבלים", "מקבלות", "קבלת", "קבלה"
]

neg_words = [
    "נדחה", "נדחית", "נדחים", "נדחות", "דוחה", "דוחים",
    "דוחות", "דחיית", "דחייה"
]

titles = [
    "פסק-דין", "פסק דין", "החלטה",
    "פסק-דין חלקי", "פסק-דין (חלקי)", "פסק דין (חלקי)",
    "פסק-דין משלים", "פסק-דין (משלים)", "פסק דין (משלים)",
    "החלטה (בעניין המשיב 1) ופסק-דין (בעניין המשיב 2)"
]


def tag_sentiment(docs: Dict[str, List[str]]) -> Dict[str, str]:
    contents, tags = {}, {}

    for name, doc in docs.items():
        for t in titles:
            if t in doc:
                title_idx = doc.index(t)

        contents[name] = doc[title_idx+1:]

    for name, content in contents.items():
        content = list_of_words(' '.join(content))

        negative, positive = 0, 0

        for w in content:
            if w in pos_words:
                positive += 1
            elif w in neg_words:
                negative += 1

        if positive > negative:
            tags[name] = 'POSITIVE'
        elif negative > positive:
            tags[name] = 'NEGATIVE'
        else:
            tags[name] = 'NEUTRAL'

    return tags


def sentiment_prediction(docs: Dict[str, List[str]], raw_docs: Dict[str, List[str]]) -> Dict[str, int]:
    tags = tag_sentiment(raw_docs)
    X, X_valid, t, t_valid, docs_names, words_idx, tags_idx = prepare_data(docs, tags)
    model = create_sentiment_model(len(words_idx), len(tags_idx))
    train(model, X, X_valid, t, t_valid, epochs=5, batch_size=1)

    idx_words = {i: w for w, i in words_idx.items()}
    idx_words[len(idx_words)] = '<PAD>'
    idx_tags = {i : t for t, i in tags_idx.items()}

    data = np.concatenate((X, X_valid))
    targets = np.concatenate((t, t_valid))
    rv = {}

    for name, seq, tag in tqdm(zip(docs_names, data, targets), desc='Generating Predictions:'):
        preds = np.argmax(model.predict(np.array([seq]), verbose=0)[0]).item()
        rv[name] = idx_tags[preds]

    return rv


def prepare_data(docs: Dict[str, List[str]],
                 tags: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], Dict[str, int]]:
    vocab = set(reduce(lambda x, y: x+y, docs.values()))
    words_idx = {w: i for i, w in zip(range(1, len(vocab)+1), vocab)}
    tags_idx = {t: i for i, t in enumerate(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])}

    names = list(docs.keys())
    seqs = [[words_idx[w] for w in docs[name]] for name in names]
    seqs = pad_sequences(sequences=seqs)
    tg = np.array([tags_idx[tags[name]] for name in names])

    X, X_valid, t, t_valid = train_test_split(seqs, tg, names, test_size=0.25, random_state=42)

    return X, X_valid, t, t_valid, names, words_idx, tags_idx


def create_sentiment_model(vocab_size: int, num_tags: int, layer1_size: int = 2000, layer2_size: int = 100) -> Model:
    inp = L.Input((None,))
    l1 = L.Embedding(input_dim=vocab_size+1, output_dim=layer1_size)(inp)
    l2 = L.GRU(layer2_size, return_sequences=False)(l1)
    out = L.Dense(num_tags, activation='softmax')(l2)
    model = Model(inp, out)
    return model


def train(model: Sequential,
          X: np.ndarray,
          X_valid: np.ndarray,
          t: np.ndarray,
          t_valid: np.ndarray,
          epochs: int = 10,
          batch_size: int = 32
          ) -> None:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_accuracy', patience=1, mode='max')
    callbacks = [PlotLossesCallback(), es]

    model.fit(
        x=X, y=t,
        validation_data=(X_valid, t_valid),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

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

