import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch.nn.functional as F
import fnmatch
from pytorch_wavelets import DWT1DForward, DWT1DInverse

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def dwt_function(y):
    """
    :param y: audio
    :return:  y_down_k (x k downsampled and channel-wise concatenated)
    """
    dwt = DWT1DForward(J=1, mode='zero', wave='db1')
    dwt = dwt.to(y.device)

    yA, yC = dwt(y)
    yAA, yAC = dwt(yA)
    yCA, yCC = dwt(yC[0])
    yAAA, yAAC = dwt(yAA)
    yACA, yACC = dwt(yAC[0])
    yCAA, yCAC = dwt(yCA)
    yCCA, yCCC = dwt(yCC[0])

    y_down2 = torch.cat((yA, yC[0]), dim=1)
    y_down4 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)
    y_down8 = torch.cat((yAAA, yAAC[0], yACA, yACC[0], yCAA, yCAC[0], yCCA, yCCC[0]), dim=1)
    return y_down2, y_down4, y_down8

def dwt_function_single(y):
    """
    :param y: audio
    :return:  y_down_k (x k downsampled and channel-wise concatenated)
    """
    dwt = DWT1DForward(J=1, mode='zero', wave='db1')
    dwt = dwt.to(y.device)

    yA, yC = dwt(y)

    return yA, yC

def idwt_function(low, high):
    idwt = DWT1DInverse()
    idwt = idwt.to(low.device)

    y = idwt([low, high])

    return y
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files