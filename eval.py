import argparse

import itertools
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(f_true, f_predict):
    """
    Функция для оценки работы алгоритма. Метрики Accuracy и Confusion Matrix.

    Аргументы функции
    -----------------
    f_true : текстовый файл, в котором построчно записаны файл обучающей выборки MNIST и эталонная метка.
    f_predict : текстовый файл, в котором построчно записаны файл обучающей выборки MNIST и предсказанная в результате
             работы алгоритма метка.

    Функция возвращает список из двух элементов: значение accuracy (тип np.float64) и confusion matrix (тип np.ndarray).

    """
    y_tr = []
    y_pr = []
    if os.path.exists(f_true) and os.path.exists(f_predict):
        for line_tr, line_pr in zip(open(f_true, 'r'), open(f_predict, 'r')):
            if line_tr.split()[0] == line_pr.split()[0]:
                y_tr.append(line_tr.split()[1])
                y_pr.append(line_pr.split()[1])
            else:
                raise Exception('Order of files is not the same.')
    else:
        raise Exception("File does not exist.")

    y_tr, y_pr = np.asarray(y_tr), np.asarray(y_pr)
    acc = accuracy_score(y_tr, y_pr)
    conf_m = confusion_matrix(y_tr, y_pr)
    return [acc, conf_m]


parser = argparse.ArgumentParser()
parser.add_argument('test_gt', help='Path to the file with ground truth, .txt format')
parser.add_argument('test_inf', help='Path to the file with inference results, .txt format')

if __name__ == "__main__":
    arg = parser.parse_args()
    [res_acc, res_conf_m] = evaluate(arg.test_gt, arg.test_inf)
