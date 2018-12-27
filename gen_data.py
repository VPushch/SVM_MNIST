import argparse
import glob


def create_txt(mode, data_dir, txt_dir):
    """
    Создается текстовый файл, где каждая строка задает файл обучающей выборки MNIST и метку (последнее не обязательно).

    Аргументы функции
    -----------------
    mode : режим работы
        Допустимые значения 'train' и 'inference'. Значение аргумента влияет на данные, записываемые в файл txt_dir.
        Для режима 'train' в файл txt_dir построчно записывается файл обучающей выборки MNIST и соответствующая метка,
        разделитель - символ пробела.
        Для режима 'inference' в файл txt_dir построчно записывается только файл обучающей выборки MNIST.
    data_dir : директория, которая включает в себя каталоги с изображениями, разделенными по папкам
        с номером соответствующего класса в названии.
    txt_dir : путь к файлу, в который построчно будут записываться файлы обучающей выборки и метка
        (последнее не обязательно).

    """
    class_dir = [i for i in glob.glob(data_dir + '\*')]
    with open(txt_dir, 'w') as f:
        for c_dir in class_dir:
            if mode == 'train':
                f.writelines(['%s\n' % (i + r' ' + c_dir[-1]) for i in glob.glob(c_dir + '\*')])
            else:
                f.writelines(['%s\n' % i for i in glob.glob(c_dir + '\*')])


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'inference'], help='Choosing the mode to write data to the file')
parser.add_argument('data_dir', help='Directory with separated by folders images')
parser.add_argument('txt_dir', help='Path to the file where the data will be written line by line')

if __name__ == "__main__":
    arg = parser.parse_args()
    create_txt(arg.mode, arg.data_dir, arg.txt_dir)
