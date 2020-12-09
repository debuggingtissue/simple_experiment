# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from fastai.vision.all import *
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import sys
from PIL import Image, ImageDraw, ImageOps
from os import listdir
from os.path import isfile, join


def label_func(f): return f[0].isupper()


def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()


def merge_images_horizontally(list_of_image_paths):
    images = [Image.open(x) for x in list_of_image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


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


def merge_images_vertically(list_of_image_paths):
    images = [Image.open(x) for x in list_of_image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im


def save_loss_plot(learner, epochs):
    learner.recorder.plot_loss()
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.savefig(f'{epochs}_epochs_plot_loss.png')
    clear_pyplot_memory()


# Cell
@patch
@delegates(subplots)
def plot_accuracy(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names)
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    # print(metrics)
    # print(names)
    # print(axs)

    axs[0].plot(metrics[:, 2], color='#ff7f0e', label='valid')
    axs[0].set_title(names[2])
    axs[0].set_xlim(0, len(metrics[:, 2]) - 1)
    axs[0].set_xlabel("n (epoch nr. = n + 1)")
    axs[0].set_ylabel("accuracy (%)")
    axs[0].legend(loc='best')

    plt.show()

# Cell
@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'loss curves')
        if name is "accuracy":
            ax.set_ylabel("accuracy (%)")
        if i is 0:
            ax.set_ylabel("loss value")
        ax.legend(loc='best')
        ax.set_xlabel("n (epoch nr. = n + 1)")
        ax.set_xlim(0, len(metrics[:, i]) - 1)
    plt.show()


# Cell
@patch
@delegates(subplots)
def plot_metrics(self: Learner, **kwargs):
    self.recorder.plot_metrics(**kwargs)


def experiment_1a(epochs, output_directory="experiment_1a"):
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path / "images")
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    clear_pyplot_memory()

    for epoch_nr in epochs:
        # print(learn.opt)
        # print(learn.wd)
        # print(learn.moms)
        # print(learn.lr)
        # print(learn.opt_func)
        # print(learn.opt.hypers)

        learn = cnn_learner(dls, resnet18, metrics=[accuracy])

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        learn.fit(epoch_nr)
        save_plots(learn, epoch_nr, output_directory)


def experiment_1b(epochs, output_directory="experiment_1b"):
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path / "images")
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    clear_pyplot_memory()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])

        learn.fine_tune(epoch_nr)
        save_plots(learn, epoch_nr, output_directory)


def experiment_2a(epochs, output_directory="experiment_2a"):
    path = "dataset_density/"
    files = get_image_files("dataset_density")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224))
    dls = data_block.dataloaders(Path(path), bs=10)

    # print(learn.opt)
    # print(learn.wd)
    # print(learn.moms)
    # print(learn.lr)
    # print(learn.opt_func)
    # print(learn.opt.hypers)


    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])
        learn.fit(epoch_nr)
        save_plots(learn, epoch_nr, output_directory)


def experiment_2b(epochs, output_directory="experiment_2b"):
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_density/"
    files = get_image_files("dataset_density")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224))
    dls = data_block.dataloaders(Path(path), bs=10)
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    clear_pyplot_memory()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])
        learn.fine_tune(epoch_nr)
        save_plots(learn, epoch_nr, output_directory)


def experiment_3a(epochs, output_directory="experiment_3a"):
    path = "dataset_SPOP_balanced_subset/"
    files = get_image_files("dataset_SPOP_balanced_subset")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224))
    dls = data_block.dataloaders(Path(path), bs=10)

    # print(learn.opt)
    # print(learn.wd)
    # print(learn.moms)
    # print(learn.lr)
    # print(learn.opt_func)
    # print(learn.opt.hypers)


    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])
        learn.fit(epoch_nr)
        save_plots(learn, epoch_nr, output_directory)

def experiment_3b(epochs, output_directory="experiment_3b"):
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_SPOP_balanced_subset/"
    files = get_image_files("dataset_SPOP_balanced_subset")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224))
    dls = data_block.dataloaders(Path(path), bs=10)
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    clear_pyplot_memory()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])
        learn.fine_tune(epoch_nr)
        save_plots(learn, epoch_nr, output_directory)



def save_plots(learn, epoch_nr, output_directory):
    learn.recorder.plot_accuracy(nrows=1, ncols=1)
    plt.savefig(f'{output_directory}/{epoch_nr}_epochs_acc.png')
    clear_pyplot_memory()

    learn.recorder.plot_loss()
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.savefig(f'{output_directory}/{epoch_nr}_epochs_loss.png')
    clear_pyplot_memory()

    learn.recorder.plot_metrics()
    plt.subplots_adjust(right=0.88)
    plt.text(0.89, 0.5, f'epochs: {epoch_nr}', fontsize=12, transform=plt.gcf().transFigure)
    plt.savefig(f'{output_directory}/{epoch_nr}_epochs_loss_and_acc.png')
    clear_pyplot_memory()

    # metrics_image_files = listdir("metrics")
    # metrics_image_file_paths = []
    # for image_file in metrics_image_files:
    #     metrics_image_file_paths.append("metrics/" + image_file)
    # metrics_image_file_paths.sort()
    #
    # metrics_overview_image = merge_images_vertically(metrics_image_file_paths)
    # metrics_overview_image.save("metrics_overview_image" + ".png", 'PNG')

    # img = PILImage.create("cid=TCGA-CH-5763-01Z-00-DX1.7d4eff47-8d99-41d4-87f0-163b2cb034bf###rl=0###x=95204###y=24800###w=800###h=800###pnc=171.png")
    # x, = first(dls.test_dl([img]))
    #
    # class Hook():
    #     def __init__(self, m):
    #         self.hook = m.register_forward_hook(self.hook_func)
    #     def hook_func(self, m, i, o): self.stored = o.detach().clone()
    #     def __enter__(self, *args): return self
    #     def __exit__(self, *args): self.hook.remove()
    #
    # with Hook(learn.model[0]) as hook:
    #     with torch.no_grad(): output = learn.model.eval()(x)
    #     act = hook.stored
    #
    # cam_map = torch.einsum('ck,kij->cij', learn.model[1][-1].weight, act[0])
    #
    # x_dec = TensorImage(dls.train.decode((x,))[0][0])
    # _, ax = plt.subplots()
    # x_dec.show(ctx=ax)
    # plt.title("last layer activation")
    # print(len(cam_map))
    #
    # cls = 0
    # ax.imshow(cam_map[cls].detach().cpu(), alpha=0.6, extent=(0, 224, 224, 0),
    #           interpolation='bilinear', cmap='magma');
    # plt.savefig(f'layer_last_activation_plot.png')
    # clear_pyplot_memory()
    #
    # print(F.softmax(output, dim=-1))
    # print(dls.vocab)
    # # print(x.shape)
    # # print(learn.model[1][-1].weight.shape)
    # # print(act.shape)
    # # cam_map.shape
    # print(learn.model[0])
    # class HookBwd():
    #     def __init__(self, m):
    #         self.hook = m.register_backward_hook(self.hook_func)
    #     def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    #     def __enter__(self, *args): return self
    #     def __exit__(self, *args): self.hook.remove()
    #
    # for layer_nr in [0, 4, 5, 6, 7]:
    #     print(learn.model[0][layer_nr])
    #     with HookBwd(learn.model[0][layer_nr]) as hookg:
    #         with Hook(learn.model[0][layer_nr]) as hook:
    #             output = learn.model.eval()(x)
    #             act = hook.stored
    #         output[0, cls].backward()
    #         grad = hookg.stored
    #
    #     w = grad[0].mean(dim=[1, 2], keepdim=True)
    #     cam_map = (w * act[0]).sum(0)
    #
    #     _, ax = plt.subplots()
    #     x_dec.show(ctx=ax)
    #     ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0, 224, 224, 0),
    #               interpolation='bilinear', cmap='magma');
    #     plt.savefig(f'layer_{layer_nr}_activation_plot.png')
    #     clear_pyplot_memory()

    # with HookBwd(learn.model[0][-5]) as hookg:
    #     with Hook(learn.model[0][-5]) as hook:
    #         output = learn.model.eval()(x)
    #         act = hook.stored
    #     output[0, cls].backward()
    #     grad = hookg.stored
    #
    # w = grad[0].mean(dim=[1, 2], keepdim=True)
    # cam_map = (w * act[0]).sum(0)
    #
    # _, ax = plt.subplots()
    # x_dec.show(ctx=ax)
    # ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0, 224, 224, 0),
    #           interpolation='bilinear', cmap='magma')
    # plt.savefig(f'other_layer_activation_plot.png')
    # clear_pyplot_memory()


if __name__ == '__main__':
    # experiment_1a([5, 10, 30, 50, 100, 500], "e_1a")
    experiment_1b([5, 10, 30, 50, 100, 500], "e_1b")
    experiment_2a([5, 10, 30, 50, 100, 500], "e_2a")
    experiment_2b([5, 10, 30, 50, 100, 500], "e_2b")

    # experiment_2b([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
