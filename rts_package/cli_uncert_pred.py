import click
import glob
import numpy as np
import os
import pathlib
import sys
import tifffile as tiff
import torch
from rich import traceback, print
from rts_package.models.unet import U2NET
from torchvision.datasets.utils import download_url
from urllib.error import URLError

from rts_package.utils import monte_carlo_dropout_proc

WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('-m', '--model', type=str,
              help='Path to an already trained XGBoost model. If not passed a default model will be loaded.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-s/-ns', '--sanitize/--no-sanitize', type=bool, default=False,
              help='Whether to remove model after prediction or not.')
@click.option('-suf', '--suffix', type=str, help='Path to write the output to')
@click.option('-o', '--output', default="", required=True, type=str, help='Path to write the output to')
@click.option('-t', '--iter', default=10, required=True, type=int, help='Number of MC-Dropout interations')
def main(input: str, suffix: str, model: str, cuda: bool, output: str, sanitize: bool, iter: int):
    """Command-line interface for rts-pred-uncert"""

    print(r"""[bold blue]
        rts-pred-uncert
        """)

    print('[bold blue]Run [green]rts-pred-uncert --help [blue]for an overview of all commands\n')
    if not model:
        model = get_pytorch_model(os.path.join(f'{os.getcwd()}', "models", "model.ckpt"))
    else:
        model = get_pytorch_model(model)
    if cuda:
        model.cuda()
    
    print('[bold blue] Calculating prediction uncertainty via MC-Dropout')
    print('[bold blue] Parsing data...')
    if os.path.isdir(input):
        input_list = glob.glob(os.path.join(input, "*"))
        for inputs in input_list:
            print(f'[bold yellow] Input: {inputs}')
            file_uncert(inputs, model, inputs.replace(input, output).replace(".tif", suffix), mc_dropout_it=iter)

    else:
        file_uncert(input, model, output)
    if sanitize:
        os.remove(os.path.join(f'{WD}', "models", "model.ckpt"))


def file_uncert(input, model, output, mc_dropout_it=10):
    input_data = read_input_data(input)
    pred_std = prediction_std(model, input_data, t=mc_dropout_it)
    
    print(f'[bold green] Output: {output}_uncert_')
    
    #write_results(pred_std, output + "_uncert_")
    write_ome_out(input_data, pred_std, output + "_uncert_")


def prediction_std(net, img, t=10):
    """
    TODO
    """
    
    net.eval()

    img = img[0, :, :]
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0)).float()

    pred_std = monte_carlo_dropout_proc(net, img, T=t)
    pred_std = pred_std.detach().cpu().numpy().astype(np.float32)

    return pred_std


def read_input_data(path_to_input_data: str):
    """
    Reads the data of an input image
    :param path_to_input_data: Path to the input data file
    """
    return tiff.imread(path_to_input_data)


def write_results(results_array: np.ndarray, path_to_write_to) -> None:
    """
    Writes the output into a file.
    :param results_array: output as a numpy array
    :param path_to_write_to: Output path
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    np.save(path_to_write_to, results_array)
    pass


def write_ome_out(input_data, results_array, path_to_write_to) -> None:
    """
    TODO
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    
    #print("write_ome_out input: " + str(input_data.shape))
    #print("write_ome_out output: " + str(results_array.shape))

    full_image = np.zeros((512, 512, 2))
    full_image[:, :, 0] = input_data[0, :, :]
    full_image[:, :, 1] = results_array
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join(path_to_write_to + ".ome.tif")) as tif_file:
        tif_file.write(full_image, photometric='minisblack', metadata={'axes': 'CYX', 'Channel': {'Name': ["image", "uncert_map"]}})
    
    pass


def get_pytorch_model(path_to_pytorch_model: str):
    """
    Fetches the model of choice and creates a booster from it.
    :param path_to_pytorch_model: Path to the Pytorch model1
    """
    download(path_to_pytorch_model)
    model = U2NET.load_from_checkpoint(path_to_pytorch_model, num_classes=5, len_test_set=120, strict=False).to('cpu')
    model.eval()
    return model


def _check_exists(filepath) -> bool:
    return os.path.exists(filepath)


def download(filepath) -> None:
    """Download the model if it doesn't exist in processed_folder already."""

    if _check_exists(filepath):
        return
    mirrors = [
        'https://zenodo.org/record/',
    ]
    resources = [
        ("mark1-PHDFM-u2net-model.ckpt", "6937290/files/mark1-PHDFM-u2net-model.ckpt", "5dd5d425afb4b17444cb31b1343f23dc"),
    ]
    # download files
    for filename, uniqueID, md5 in resources:
        for mirror in mirrors:
            url = "{}{}".format(mirror, uniqueID)
            try:
                print("Downloading {}".format(url))
                download_url(
                    url, root=str(pathlib.Path(filepath).parent.absolute()),
                    filename=filename,
                    md5=md5
                )
            except URLError as error:
                print(
                    "Failed to download (trying next):\n{}".format(error)
                )
                continue
            finally:
                print()
            break
        else:
            raise RuntimeError("Error downloading {}".format(filename))
    print('Done!')


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
