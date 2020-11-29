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
    path = "dataset/"
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
    learn.fine_tune(3)
    learn.recorder.plot_loss()
    plt.savefig(f'plot_loss.png')
    clear_pyplot_memory()

    learn.predict(files[0])
    learn.show_results()
    plt.savefig(f'plot_predictions.png')
    clear_pyplot_memory()

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(5, nrows=1)
    plt.savefig(f'plot_top_losses.png')
    clear_pyplot_memory()

    img = PILImage.create("cid=TCGA-CH-5763-01Z-00-DX1.7d4eff47-8d99-41d4-87f0-163b2cb034bf###rl=0###x=95204###y=24800###w=800###h=800###pnc=171.png")
    x, = first(dls.test_dl([img]))

    class Hook():
        def __init__(self, m):
            self.hook = m.register_forward_hook(self.hook_func)
        def hook_func(self, m, i, o): self.stored = o.detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()

    with Hook(learn.model[0]) as hook:
        with torch.no_grad(): output = learn.model.eval()(x)
        act = hook.stored

    cam_map = torch.einsum('ck,kij->cij', learn.model[1][-1].weight, act[0])


    x_dec = TensorImage(dls.train.decode((x,))[0][0])
    _, ax = plt.subplots()
    x_dec.show(ctx=ax)
    plt.title("last layer activation")
    print(len(cam_map))
    ax.imshow(cam_map[0].detach().cpu(), alpha=0.6, extent=(0, 224, 224, 0),
              interpolation='bilinear', cmap='magma');
    plt.savefig(f'layer_last_activation_plot.png')
    clear_pyplot_memory()

    print(F.softmax(output, dim=-1))
    print(dls.vocab)
    # print(x.shape)
    # print(learn.model[1][-1].weight.shape)
    # print(act.shape)
    # cam_map.shape
    print(learn.model[0])
    class HookBwd():
        def __init__(self, m):
            self.hook = m.register_backward_hook(self.hook_func)
        def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()

    for layer_nr in [0, 4, 5, 6, 7]:
        cls = 0
        print(learn.model[0][layer_nr])
        with HookBwd(learn.model[0][layer_nr]) as hookg:
            with Hook(learn.model[0][layer_nr]) as hook:
                output = learn.model.eval()(x)
                act = hook.stored
            output[0, cls].backward()
            grad = hookg.stored

        w = grad[0].mean(dim=[1, 2], keepdim=True)
        cam_map = (w * act[0]).sum(0)

        _, ax = plt.subplots()
        x_dec.show(ctx=ax)
        ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0, 224, 224, 0),
                  interpolation='bilinear', cmap='magma');
        plt.savefig(f'layer_{layer_nr}_activation_plot.png')
        clear_pyplot_memory()

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
   print("hey")
   experiment_2()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
