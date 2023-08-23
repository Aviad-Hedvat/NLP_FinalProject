from typing import Dict, Tuple, List

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from livelossplot.inputs.tf_keras import PlotLossesCallback

from keras import Sequential
from keras.layers import InputLayer, SpatialDropout1D, Bidirectional, LSTM, Embedding, Dense
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping


def data_preparation(sentences_tags: Dict) -> Tuple:
    words = [word for sentence in sentences_tags['Sentences'] for word in sentence]
    tags = [t for tag in sentences_tags['Tags'] for t in tag]

    words_idx = {w: i for i, w in enumerate(set(words))}
    tag_idx = {t: i for i, t in enumerate(set(tags))}

    sens_seqs = [[words_idx[w] for w in words[i:i+50]] for i in range(0, len(words), 50)]
    sens_sequences = pad_sequences(sequences=sens_seqs, padding='post', value=len(sens_seqs))

    tags_seqs = [[tag_idx[t] for t in tags[i:i+50]] for i in range(0, len(tags), 50)]
    tags_sequences = pad_sequences(sequences=tags_seqs, padding='post', value=tag_idx['O'])

    X_sens, X_valid_sens, t_sens, t_valid_sens = train_test_split(
        sens_sequences, tags_sequences, test_size=0.25, random_state=42)

    return X_sens, X_valid_sens, t_sens, t_valid_sens, words_idx, tag_idx


def create_model(input_size: int, vocab_size: int, tags_quantity: int) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_size))
    model.add(Embedding(input_dim=vocab_size+1, output_dim=input_size, input_length=input_size))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
    model.add(Dense(tags_quantity, activation='softmax'))

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


def ner_predictions(
        model: Sequential,
        X: np.ndarray,
        X_valid: np.ndarray,
        t: np.ndarray,
        t_valid: np.ndarray,
        words_idx: Dict,
        tags_idx: Dict
) -> List:

    idx_words = {i : w for w, i in words_idx.items()}
    idx_tags = {i: t for t, i in tags_idx.items()}

    idx_words[len(idx_words)] = '<PAD>'

    data = np.concatenate((X, X_valid))
    tags = np.concatenate((t, t_valid))
    res = []

    for d_seq, t_seq in tqdm(zip(data, tags), desc='Generating Predictions:'):
        preds = np.argmax(model.predict(np.array([d_seq]), verbose=0)[0], axis=-1)
        res.append([(idx_words[wi], idx_tags[ti], idx_tags[pi]) for wi, ti, pi in zip(d_seq, t_seq, preds)])

    return res




