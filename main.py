import argparse
import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC


class SVMDigits:
    """
    Скрипт для обучения и применения SVM модели для задачи распознавания рукописного текста на примере базы MNIST.

    Аргументы конструктора
    -----------------
    mode : режим работы
        Допустимые значения 'train' и 'inference'.
        Для режима 'train' проиводится обучение модели на выборке из файла data_dir,
        полученная модель сохраняется в model_dir.
        Для режима 'inference' производится загрузка моели из model_dir, модель применяется к файлам-изображениям
        из data_dir, результаты предсказания сохраняются в save_inf
    data_dir : текстовый файл с построчно перечисленными файлами-изображениями обучающего набора данных и метки класса
        (для режима "train") или без метки (для режима "inference"), .txt формат, тип str
    model_dir : путь к файлу для сохранения или загрузки модели, .sav формат, тип str.
    save_inf: путь к файлу для сохранения результата предскаания модели, .txt формат, тип str.
    preprocess: флаг для запуска предварительной обработки данных перед обучением модели или классификацией
        на основе обученной ранее, тип bool.

    """
    def __init__(self, mode, data_dir, model_dir, save_inf=None, preprocess=False):
        self.mode = mode
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.save_inf = save_inf
        self.preprocess = preprocess

        self.H = 28
        self.W = 28
        self.GAMMA = 0.05
        self.C = 5

        self.instance_count = sum(1 for line in open(self.data_dir, 'r'))
        self.X, self.y, self.im_dir = self.load_data()

        if self.mode == 'train':
            self.model = self.train()
        else:
            self.model = self.inference()
        print(self.model)

    def load_data(self):
        """Загрузка данных в соответствии с режимом работы скрипта"""
        print('Data loading')
        X = np.zeros((self.instance_count, self.H * self.W))
        y = np.zeros(self.instance_count)
        im_dir = []

        for i, line in zip(range(self.instance_count), open(self.data_dir, 'r')):
            if self.mode == 'train':
                im, label = line.split()
            else:
                im, label = line[:-1], 999
                im_dir.append(line[:-1])

            if self.preprocess:
                print('Image preprocessing')
                im = cv2.imread(im)[:, :, 0]
                im = self.deskew(im)
                X[i], y[i] = np.reshape(im, -1), int(label)
                X[i] = X[i]/255
            else:
                X[i], y[i] = np.reshape(cv2.imread(im)[:, :, 0], -1), int(label)

        return X, y, im_dir

    def train(self):
        """Обучение классификатора и сохранение полученной модели"""
        print('Train')
        svm_clf = SVC(kernel='rbf', gamma=self.GAMMA, C=self.C, decision_function_shape='ovo')
        svm_clf.fit(self.X, self.y)
        pickle.dump(svm_clf, open(self.model_dir, 'wb'))
        return svm_clf

    def inference(self):
        """Применение загруженной классификационной модели"""
        if isinstance(self.save_inf, str) and os.path.exists(self.model_dir):
            print('Inference')
            svm_clf = pickle.load(open(self.model_dir, 'rb'))
            res = svm_clf.predict(self.X)
            with open(self.save_inf, 'w') as f:
                f.writelines(['%s \n' % (line + ' ' + str(int(r))) for r, line in zip(res, self.im_dir)])
        else:
            raise Exception('The file for loading the model is not specified or the specified file does not exist.')
        return svm_clf

    def deskew(self, im):
        """Устранение наклонов цифр на изображении"""
        moments = cv2.moments(im)
        if abs(moments['mu02']) < 1e-2:
            return im
        skew = moments['mu11'] / moments['mu02']  # covariance(x,y) / var(y^2)
        tr_matrix = np.float32([[1, skew, -0.5*self.H*skew], [0, 1, 0]])
        im = cv2.warpAffine(im, tr_matrix, (self.H, self.W), flags=cv2.WARP_INVERSE_MAP or cv2.INTER_LINEAR)
        return im


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'inference'], help='Choosing the mode to write data to the file')
parser.add_argument('--data', required=True, help='Path to the file where the data is written line by line, '
                                                  '.txt format')
parser.add_argument('--model', required=True, help='Directory to save or download model, .sav format.')
parser.add_argument('--inf', nargs='?', default=None, help='Directory to save inference results, .txt format.')
parser.add_argument('--prepr', const=True, nargs='?', help='Options for image preprocessing')

if __name__ == '__main__':
    arg = parser.parse_args()
    print(arg)
    SVMDigits(arg.mode,
              arg.data,
              arg.model,
              arg.inf,
              arg.prepr)
