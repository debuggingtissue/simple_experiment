# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from fastai.vision.all import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def label_func(f): return f[0].isupper()

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()

def experiment_1():
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path / "images")
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    clear_pyplot_memory()

    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(10)
    learn.recorder.plot_loss()
    plt.savefig(f'plot_loss.png')
    clear_pyplot_memory()

    learn.predict(files[0])
    learn.show_results()
    plt.savefig(f'plot_predictions.png')
    clear_pyplot_memory()

def experiment_2():
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset"
    files = get_image_files("dataset")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224))
    dls = data_block.dataloaders(Path(path), bs=10)
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    clear_pyplot_memory()

    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(10)
    learn.recorder.plot_loss()
    plt.savefig(f'plot_loss.png')
    clear_pyplot_memory()

    learn.predict(files[0])
    learn.show_results()
    plt.savefig(f'plot_predictions.png')
    clear_pyplot_memory()


if __name__ == '__main__':
   print("hey")
   experiment_2()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
