import time
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def message(config, string, time_stamp=True):
    t = time.localtime()
    if time_stamp:
        prefix = f"[{str(t.tm_hour).zfill(2)}:{str(t.tm_min).zfill(2)}:{str(t.tm_sec).zfill(2)}] "
    else:
        prefix = " "*11

    message = prefix + string

    _print(config, message)
    
def line(config):
    _print(config, "-------------------------------------\n")

def config_init(config):
    if config.model_path:
        file_path, file_name = os.path.split(config.model_path)
        config.results_dir = file_path
        config.log_path = os.path.join(config.results_dir, "out.debug.log")

    if config.out_to_folder == "True":
        t = int(time.time())
        if 'results' not in os.listdir():
            os.mkdir('results')
        results_dir = os.path.join('results', str(t))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        config.results_dir = results_dir

    _print(config, config)
    line(config)

    return config

def _print(config, message):
    if config.out_to_folder == "True":
        with open(os.path.join(config.results_dir, "out.debug.log"), "a+") as f:
            terminal = sys.stdout
            sys.stdout = f
            print(message)
            sys.stdout = terminal

    print(message)

def show_out (image, name):
    image = image.data.cpu().numpy()[0]

    if image.shape[0] == 2:
        plt.imshow(image[1], cmap='gray')
        plt.savefig(name+'.png')
        plt.show()
    else:
        plt.imshow(image.reshape((256, 256)),cmap='gray')
        plt.savefig("out/" + name + '.png')
        plt.show()

def evaluate_error(out, target):
    errors = {'DIC': 0, 'JSC': 0}

    out = out.cpu().numpy()
    out = np.around(out[:, 1, ...])
    target = target.cpu().numpy()

    tp = np.sum(out[out == target == 1], axis=(2, 3))
    fp = np.sum(out[out == target == 0], axis=(2, 3))
    tn = np.sum(out[out == 1] - tp, axis=(2, 3))
    fn = np.sum(out[out == 0] - fp, axis=(2, 3))

    errors['DIC'] = np.sum(2*tp / (2*tp + fn + fp))
    errors['JSC'] = np.sum((tp) / (tp + fp + fn))

    return errors
