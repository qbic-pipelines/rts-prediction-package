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
def main(input: str, suffix: str, model: str, cuda: bool, output: str, sanitize: bool):
    """Command-line interface for rts_package"""

    print(r"""[bold blue]
        rts_package
        """)

    print('[bold blue]Run [green]rts_package --help [blue]for an overview of all commands\n')
    if not model:
        model = get_pytorch_model(os.path.join(f'{os.getcwd()}', "models", "model.ckpt"))
    else:
        model = get_pytorch_model(model)
    if cuda:
        model.cuda()
    print('[bold blue] Parsing data')
    if os.path.isdir(input):
        input_list = glob.glob(os.path.join(input, "*"))
        for inputs in input_list:
            file_prediction(inputs, model, inputs.replace(input, output).replace(".tif", suffix))
            file_uncert(inputs, model, inputs.replace(input, output).replace(".tif", suffix))

    else:
        file_prediction(input, model, output)
    if sanitize:
        os.remove(os.path.join(f'{WD}', "models", "model.ckpt"))


def file_prediction(input, model, output):
    data_to_predict = read_data_to_predict(input)
    print('[bold blue] Performing predictions')
    predictions = predict(data_to_predict, model)
    print(f'[bold blue]Writing predictions to {output}')
    #write_results(predictions, output)
    write_ome_out(data_to_predict, predictions, output)


def read_data_to_predict(path_to_data_to_predict: str):
    """
    Parses the data to predict and returns a full Dataset include the DMatrix
    :param path_to_data_to_predict: Path to the data on which predictions should be performed on
    """
    return tiff.imread(path_to_data_to_predict)


def write_results(predictions: np.ndarray, path_to_write_to) -> None:
    """
    Writes the predictions into a human readable file.
    :param predictions: Predictions as a numpy array
    :param path_to_write_to: Path to write the predictions to
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    np.save(path_to_write_to, predictions)
    pass

def write_ome_out(data_to_predict, predictions, path_to_write_to) -> None:
    """
    Writes the predictions into a human readable file.
    :param predictions: Predictions as a numpy array
    :param path_to_write_to: Path to write the predictions to
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    
    print("in shape: " + str(data_to_predict.shape))
    print("out shape: " + str(predictions.shape))

    full_image = np.zeros((512, 512, 2))
    full_image[:, :, 0] = data_to_predict[0, :, :]
    full_image[:, :, 1] = predictions
    full_image = np.transpose(full_image, (2, 0, 1))
    with tiff.TiffWriter(os.path.join(path_to_write_to + ".ome.tif")) as tif_file:
        tif_file.write(full_image, photometric='minisblack', metadata={'axes': 'CYX', 'Channel': {'Name': ["image", "seg_mask"]}})
    
    pass

def predict(data_to_predict, model):
    img = data_to_predict[0, :, :]
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0)).float()
    logits = model(img)[0]
    prediction = torch.argmax(logits.squeeze(), dim=0).cpu().detach().numpy().squeeze()
    return prediction

def file_uncert(input, model, output):
    data_to_predict = read_data_to_predict(input)
    print('[bold blue] uncert...')
    pred_std = prediction_std(model, data_to_predict, t=4)
    print(f'[bold blue]Writing out to {output}')
    
    write_ome_out(data_to_predict, pred_std, output + "_uncert_")

def prediction_std(net, img, t):
    """
    TODO
    """
    
    net.eval()

    img = img[0, :, :]
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0)).float()

    pred_std = monte_carlo_dropout_proc(net, img, T=t)
    pred_std = pred_std.detach().cpu().numpy().astype(np.float32)

    return pred_std


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
        ("model.ckpt", "5181261/files/model.ckpt", "f73c3d232fd1d1eae5547547b37ed4f1"),
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
