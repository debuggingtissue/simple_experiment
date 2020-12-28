# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from fastai.vision.all import *
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from utils import image_merger
from PIL import Image, ImageDraw, ImageOps
import os
from os.path import isfile, join
import re
from pathlib import *
from fastai.learner import load_learner


def save_plots(learn, epoch_nr, output_directory, dls=None):
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

    dls.show_batch()
    plt.savefig(f'{output_directory}/show_batch.png')
    clear_pyplot_memory()


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

def generate_data_block(self):
    dat_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                          get_items=get_image_files,
                          splitter=RandomSplitter(),
                          get_y=parent_label,
                          item_tfms=self.transforms["item"],
                          batch_tfms=self.transforms["batch"])

    return data_block


def dataloaders(self, dataset_path, batch_size):
    dls = self.data_block.dataloaders(Path(dataset_path), bs=batch_size)
    return dls


def create_experiment_images(experiment_result_directory_paths, output_directory):
    for experiment_result_directory_path in experiment_result_directory_paths:
        image_merger.ImageMerger.create_overview_image(experiment_result_directory_path, output_directory)


def experiment_1a(epochs, output_directory="experiment_1a"):
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path / "images")
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

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
        save_plots(learn, epoch_nr, output_directory, dls)
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))




def experiment_1b(epochs, output_directory="experiment_1b"):
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path / "images")
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])

        learn.fine_tune(epoch_nr)
        save_plots(learn, epoch_nr, output_directory, dls)
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))

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
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))



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
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))



def experiment_2c(epochs, output_directory="experiment_2c"):
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_density_balanced_35/"
    files = get_image_files("dataset_density_balanced_35")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224))
    dls = data_block.dataloaders(Path(path), bs=10)
    dls.show_batch()
    # plt.savefig(f'show_batch.png')
    # clear_pyplot_memory()
    #
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    #
    # for epoch_nr in epochs:
    #     learn = cnn_learner(dls, resnet18, metrics=[accuracy])
    #     learn.fine_tune(epoch_nr)
    #     save_plots(learn, epoch_nr, output_directory, dls)
    #     learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))



# def experiment_3a(epochs, output_directory="experiment_3a"):
#     path = "dataset_SPOP_random_balanced_46/"
#     files = get_image_files("dataset_SPOP_random_balanced_46")
#
#     data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
#                            get_items=get_image_files,
#                            splitter=RandomSplitter(),
#                            get_y=parent_label,
#                            item_tfms=Resize(224))
#     dls = data_block.dataloaders(Path(path), bs=10)
#
#     # print(learn.opt)
#     # print(learn.wd)
#     # print(learn.moms)
#     # print(learn.lr)
#     # print(learn.opt_func)
#     # print(learn.opt.hypers)
#
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     for epoch_nr in epochs:
#         learn = cnn_learner(dls, resnet18, metrics=[accuracy])
#         learn.fit(epoch_nr)
#         save_plots(learn, epoch_nr, output_directory)

def experiment_3a(epochs, output_directory="experiment_3a"):
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_SPOP_random_balanced_46/"
    files = get_image_files("dataset_SPOP_random_balanced_46")

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
        save_plots(learn, epoch_nr, output_directory, dls)
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))



def experiment_3b(epochs, output_directory="experiment_3b"):
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_SPOP_v2_random_balanced_35/"
    files = get_image_files("dataset_SPOP_v2_random_balanced_35")

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
        save_plots(learn, epoch_nr, output_directory, dls)
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))

def experiment_3b(epochs, output_directory="experiment_3b"):
    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_SPOP_v2_random_balanced_35/"
    files = get_image_files("dataset_SPOP_v2_random_balanced_35")

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
        save_plots(learn, epoch_nr, output_directory, dls)
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))


def experiment_3c(output_directory="experiment_3c"):
    # timg = TensorImage(array(img)).permute(2, 0, 1).float() / 255.
    # def _batch_ex(bs): return TensorImage(timg[None].expand(bs, *timg.shape).clone())
    #
    # tfms = aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=10., min_zoom=1, max_zoom=1, max_lighting=0.5, max_warp=0, p_affine=0.7,
    #                       p_lighting=0.2, xtra_tfms=None, size=None, mode='bilinear', pad_mode='zeros', align_corners=True, batch=False, min_scale=1.0)
    # y = _batch_ex(9)
    # for t in tfms: y = t(y, split_idx=0)
    # _, axs = plt.subplots(1, 3, figsize=(12, 3))
    # for i, ax in enumerate(axs.flatten()):
    #     show_image(y[i], ctx=ax)
    # plt.show()

    #_pad_modes = {'zeros': 'constant', 'border': 'edge', 'reflection': 'reflect'}


    # path = untar_data(URLs.PETS)
    # print(path.ls())
    path = "dataset_SPOP_v2_random_balanced_35/"
    files = get_image_files("dataset_SPOP_v2_random_balanced_35")

    data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           get_items=get_image_files,
                           splitter=RandomSplitter(),
                           get_y=parent_label,
                           item_tfms=Resize(224),
                           batch_tfms=aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=10., min_zoom=1, max_zoom=1, max_lighting=0.5, max_warp=0, p_affine=0.7,
                                                     p_lighting=0.2, xtra_tfms=None, size=None, mode='bilinear', pad_mode='zeros', align_corners=True, batch=False, min_scale=1.0))

    dls = data_block.dataloaders(Path(path), bs=10)
    dls.show_batch()
    plt.savefig(f'show_batch.png')
    plt.show()
    clear_pyplot_memory()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for epoch_nr in epochs:
        learn = cnn_learner(dls, resnet18, metrics=[accuracy])
        learn.fine_tune(epoch_nr)
        save_plots(learn, epoch_nr, output_directory, dls)
        learn.export(os.path.abspath(output_directory + f"/{epoch_nr}_export.pkl"))




def create_image_array_from_directory_path(path):
    path = Path(path)
    file_paths = [PILImage.create(p) for p in path.iterdir() if p.is_file()]
    print(file_paths)
    return file_paths

def experiment_1b_inference():
    print("yolo")
    #learn = cnn_learner(dls, resnet18, metrics=[accuracy])
    learn = load_learner("30_export.pkl")
    cats = create_image_array_from_directory_path("cat_dog_test_set/dogs")
    dl = learn.dls.test_dl(cats)
    op = learn.get_preds(dl=dl)
    print(op)


if __name__ == '__main__':
    # experiment_1b([30], "e_1b")
    # experiment_1a([50], "e_1a")
    experiment_3c([5, 10, 30, 50, 100, 500], "e_3c")
    # experiment_2a([5, 10, 30, 50, 100, 500], "e_2a")
    # # experiment_2b([5, 10, 30, 50, 100, 500], "e_2b")
    # experiment_2c([5, 10, 30, 50, 70, 100, 500], "e_2c")
    # experiment_3a([5, 10, 30, 50, 70, 100, 500 ], "e_3a")
    #experiment_3b([1,2], "save_model_test")
    # image_merger.ImageMerger.create_overview_image("e_2c", "lol")
    #create_experiment_images(
    #    ["e_1a_thesis", "e_1b_thesis", "e_2a_thesis", "e_2b_thesis", "e_2c_thesis", "e_3a_thesis", "e_3b_thesis"], "overview_images")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
