from typing import Dict, List, Tuple
from collections import OrderedDict
from functools import reduce
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from keras.utils import pad_sequences
from keras import layers as L
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from livelossplot.inputs.tf_keras import PlotLossesCallback


def keywords_extraction(
        docs: Dict[str, List[str]],
        mode: int = 1,
        top_keywords: int = 5
) -> Dict[str, OrderedDict[str, float]]:
    """

    :param docs:
    :param mode: 1 for tf-idf, 2 for word2vec, 3 for autoencoder, 1 is default
    :param top_keywords:
    :return:
    """

    if mode == 1:
        scores = tfidf_scores(docs)
    elif mode == 2:
        scores = word2vec_scores(docs)
    elif mode == 3:
        scores = autoencoder_scores(docs)
    else:
        raise ValueError(f'argument mode has to be one of [1, 2 ,3] instead of {mode}')

    for name, score in scores.items():
        vals = sorted(score.items(), key=lambda x: x[1], reverse=True)

        if top_keywords is not None:
            vals = vals[:top_keywords]

        scores[name] = OrderedDict(vals)

    return scores


def tfidf_scores(docs: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    tf, df, scores = {}, {}, {}

    for name, doc in docs.items():
        tf[name] = {}

        for word in doc:
            tf[name][word] = tf[name].get(word, 0) + 1

        for word in tf[name].keys():
            df[word] = df.get(word, 0) + 1

    for name, dtf in tf.items():
        scores[name] = {}

        for word, wtf in dtf.items():
            scores[name][word] = wtf * np.log(len(docs) / df[word])

    return scores


def word2vec_scores(docs: Dict[str, List[str]], epochs: int = 30, size: int = 400) -> Dict[str, Dict[str, float]]:
    model = Word2Vec(sentences=docs.values(), epochs=epochs, vector_size=size, min_count=0)
    embeddings = {w: model.wv[w] for w in model.wv.key_to_index}

    return generate_embeddings_scores(docs, embeddings)


def autoencoder_scores(docs: Dict[str, List[str]], epochs: int = 10, size: int = 400) -> Dict[str, Dict[str, float]]:
    X, t, words_idx = prepare_data(docs)
    autoencoder, encoder = create_encoders(len(words_idx), emb_size=size)
    train(autoencoder, X, t, X, t, epochs=epochs)

    data = np.concatenate((X, t))
    idx_words = {i: w for w, i in words_idx.items()}
    word_freqs, emb_sums = {}, {}
    data_slices = [data[i:i + 32] for i in range(0, len(data), 32)]

    for ds in tqdm(data_slices, desc='Generating Embeddings:'):
        spred = encoder.predict(np.array(ds), verbose=0)[0]

        for d, pred in zip(ds, spred):
            for w, e in zip(d, pred):
                if w != 0:
                    word = idx_words[w]
                    word_freqs[word] = word_freqs.get(word, 0) + 1
                    emb_sums[word] = emb_sums.get(word, np.zeros((size,))) + e

    embeddings = {}

    for word in emb_sums.keys():
        embeddings[word] = emb_sums[word] / word_freqs[word]

    return generate_embeddings_scores(docs, embeddings)


def prepare_data(docs: Dict[str, List[str]]) -> Tuple:
    vocab = set(reduce(lambda x, y: x + y, docs.values()))
    words_idx = {w: i for i, w in zip(range(1, len(vocab) + 1), vocab)}
    sequences = [[words_idx[w] for w in d] for d in docs.values()]

    split_seqs = [[s[i:i + 50] for i in range(0, len(s), 50)] for s in sequences]
    split_seqs = reduce(lambda x, y: x + y, split_seqs)
    data = pad_sequences(sequences=split_seqs)

    X, t = train_test_split(data, test_size=0.25, random_state=42)

    return X, t, words_idx


def create_encoders(vocab_size: int, h_size: int = 200, emb_size: int = 400) -> Tuple[Model, Model]:
    inp = L.Input((None,))
    embedding = L.Embedding(input_dim=vocab_size + 1, output_dim=h_size)(inp)
    enc_out = L.GRU(emb_size, return_sequences=True)(embedding)
    dec_in = L.GRU(emb_size, return_sequences=True)(enc_out)
    dec_h = L.Dense(h_size)(dec_in)
    dec_out = L.Dense(vocab_size + 1, activation='softmax')(dec_h)

    autoencoder = Model(inp, dec_out)
    encoder = Model(inp, enc_out)

    return autoencoder, encoder


def generate_embeddings_scores(
        docs: Dict[str, List[str]],
        embeddings: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    docs_idx = {d: i for d, i in zip(docs.keys(), range(len(docs)))}
    words_idx = {w: i for w, i in zip(embeddings.keys(), range(len(embeddings)))}
    idx_words = {i: w for w, i in words_idx.items()}

    freqs = np.zeros((len(docs), len(embeddings)), dtype=float)

    for name, doc in docs.items():
        dlen = len(doc)

        for word in doc:
            freqs[docs_idx[name], words_idx[word]] += 1 / dlen

    stack = np.stack([embeddings[idx_words[i]] for i in range(len(embeddings))])
    mat = freqs @ stack

    rv = {}

    for name in docs.keys():
        rv[name] = {}

        for word in embeddings.keys():
            rv[name][word] = np.exp(-np.sum((mat[docs_idx[name]] - stack[words_idx[word]]) ** 2))

    return rv


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


