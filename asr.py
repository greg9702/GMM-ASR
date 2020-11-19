"""
Inspired by: https://github.com/PacktPublishing/Python-Machine-Learning-Cookbook/blob/master/Chapter07/speech_recognizer.py
"""

import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from typing import Dict, List, Tuple

from hmm_trainer import HMMTrainer

TRAIN_DIR = "Data/isolated_digits_ti_train_endpt"
TEST_DIR = "Data/isolated_digits_ti_test_endpt"

NUM_CEP = 13  # parameter for MFCC
NUM_LABELS = 11  # 1, 2, 3, 4, ...
PRINT_FREQ = 500

Label = str
Features = np.ndarray
FeaturesByLabelsDict = Dict[Label, Features]
ModelLabelTuple = Tuple[HMMTrainer, Label]
ModelsList = List[ModelLabelTuple]


def extract_label(file_name: str) -> str:
    return file_name[0]


def get_test_files() -> [str]:
    test_files = []
    for gender_dir_name in os.listdir(TEST_DIR):
        # MAN, WOMAN
        gender_dir = os.path.join(TEST_DIR, gender_dir_name)
        if not os.path.isdir(gender_dir):
            continue

        for person_dir_name in os.listdir(gender_dir):
            # AH, AR, ...
            person_dir = os.path.join(gender_dir, person_dir_name)

            for file_name in [x for x in os.listdir(person_dir) if x.endswith('.wav')]:
                file = os.path.join(person_dir, file_name)
                test_files.append(file)
    return test_files


def obtain_mfcc_features() -> FeaturesByLabelsDict:
    mfcc_by_label: Dict[str, np.ndarray] = {}

    for gender_dir_name in os.listdir(TRAIN_DIR):
        # MAN, WOMAN
        gender_dir = os.path.join(TRAIN_DIR, gender_dir_name)
        if not os.path.isdir(gender_dir):
            continue

        for person_dir_name in os.listdir(gender_dir):
            # AH, AR, ...
            person_dir = os.path.join(gender_dir, person_dir_name)

            for file_name in [x for x in os.listdir(person_dir) if x.endswith('.wav')]:

                label = extract_label(file_name)
                if label not in mfcc_by_label:
                    mfcc_by_label[label] = np.empty((0, NUM_CEP))

                file_path = os.path.join(person_dir, file_name)
                sampling_freq, audio = wavfile.read(file_path)

                # Extract MFCC features
                mfcc_features = mfcc(audio, sampling_freq, numcep=NUM_CEP)

                mfcc_by_label[label] = np.append(mfcc_by_label[label], mfcc_features, axis=0)
    return mfcc_by_label


def train_hmm_models(data: FeaturesByLabelsDict) -> ModelsList:
    models: ModelsList = []
    for label, features in data.items():
        print('features of', label, 'shape =', features.shape)
        hmm_model = HMMTrainer()
        hmm_model.train(features)
        models.append((hmm_model, label))
    return models


def predict_label(models_list: ModelsList, features: Features) -> Label:
    max_score = -10_000
    best_label = None

    for item in models_list:
        hmm_model, label = item
        score = hmm_model.get_score(features)
        if score > max_score:
            max_score = score
            best_label = label
    return best_label


if __name__ == '__main__':
    print("COLLECTING FEATURES")
    features_by_label = obtain_mfcc_features()

    print("TRAINING MODEL")
    hmm_models = train_hmm_models(features_by_label)

    print("EVALUATING MODEL")
    test_files = get_test_files()
    no_files = len(test_files)
    no_correct_pred = 0
    for i, test_file in enumerate(test_files):
        sampling_freq, audio = wavfile.read(test_file)
        mfcc_features = mfcc(audio, sampling_freq)

        true_label = extract_label(test_file[test_file.rfind('/') + 1:])
        predicted_label = predict_label(hmm_models, mfcc_features)
        no_correct_pred += true_label == predicted_label

        if not i % PRINT_FREQ:
            print(f"File: {test_file}\nTrue label: {true_label}, predicted label: {predicted_label}\n")

    print(f"Accuracy: {no_correct_pred / no_files}")
