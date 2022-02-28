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
        file_path, filename = os.path.split(config.model_path)
        filename, _ = os.path.splitext(filename)
        config.results_dir = file_path
        config.log_path = os.path.join(config.results_dir, filename+"_test.log")
        with open(config.log_path, "w") as f:
            f.write("")

    elif config.out_to_folder == "True":
        t = int(time.time())
        if 'results' not in os.listdir():
            os.mkdir('results')
        results_dir = os.path.join('results', str(t))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        config.results_dir = results_dir
        config.log_path = os.path.join(config.results_dir, "out.debug.log")

    # _print(config, config)
    # line(config)

    return config

def _print(config, message):
    if config.out_to_folder == "True":
        with open(os.path.join(config.results_dir, "out.debug.log"), "a+") as f:
            terminal = sys.stdout
            sys.stdout = f
            print(message)
            sys.stdout = terminal

    print(message)

def show_out(image, name):
    image = image.data.cpu().numpy()[0]

    if image.shape[0] == 2:
        plt.imshow(image[1], cmap='gray')
        plt.savefig(name+'.png')
        plt.show()
    else:
        plt.imshow(image.reshape((256, 256)),cmap='gray')
        plt.savefig("out/" + name + '.png')
        plt.show()

def show_out_full(img_list, gt_list, out_list, path):
    length = len(img_list)
    
    for i in range(length):
        img = img_list[i].data.cpu().permute(1,2,0).squeeze().numpy()
        gt = gt_list[i].data.cpu().numpy()
        out = out_list[i][1].data.cpu().numpy()
        plt.subplot(4, 3, i*3+1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.subplot(4, 3, i*3+2)
        plt.imshow(gt, cmap='gray')
        plt.subplot(4, 3, i*3+3)
        plt.imshow(out, cmap='gray')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def evaluate_error(out, target):
    errors = {'DIC': 0, 'JSC': 0}

    out = out.data.cpu().numpy()
    out = np.around(out[:, 1, ...])
    target = target.cpu().numpy()

    true = np.full_like(out, 3)
    false = np.full_like(out, 3)
    tp = np.zeros_like(out)
    tn = np.zeros_like(out)
    fp = np.zeros_like(out)
    fn = np.zeros_like(out)
    
    true[out == target] = 1
    tp[true == out] = 1
    true -= 1
    tn[true == out] = 1

    false[out != target] = 1
    fn[false == out] = 1
    fn[false == target] = 1

    tp = np.sum(tp, axis=(1, 2))
    tn = np.sum(tn, axis=(1, 2))
    fp = np.sum(fp, axis=(1, 2))
    fn = np.sum(fn, axis=(1, 2))

    # print(tp, tn, fp, fn)

    errors['DIC'] = np.sum(2*tp / (2*tp + fn + fp))
    errors['JSC'] = np.sum((tp) / (tp + fp + fn))

    return errors
