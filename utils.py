import time
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from PIL import Image

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
    if config.model_path and config.out_to_folder == 'True':
        file_path, filename = os.path.split(config.model_path)
        filename, _ = os.path.splitext(filename)
        config.results_dir = file_path
        if config.use_adv > 0:
            config.log_path = os.path.join(config.results_dir, filename+f"_test_adv{config.use_adv}.log")
        else:
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
    o = image.data.cpu().numpy()
    # out = np.around(out[:, 1, ...])
    _, H, W = o.shape
    out = np.zeros((H, W))
    out[o[0] < o[1]] = 1
    image = out

    if image.shape[0] == 2:
        plt.imshow(image[1], cmap='gray')
        plt.savefig(name+'.png')
        plt.show()
    else:
        plt.imshow(image.reshape((256, 256)),cmap='gray')
        plt.savefig("out/" + name + '.png')
        plt.show()

def show_out_full(img_list, gt_list, out_list, path):
    image_stats = {'mean':[0.7331, 0.6158, 0.5599],
                   'std':[0.1522, 0.1724, 0.1930]}
    length = len(img_list)
    
    for i in range(4):
        img = img_list[i].data.cpu().permute(1,2,0).squeeze().numpy()
        gt = gt_list[i].data.cpu().numpy()
        # out = out_list[i][1].data.cpu().numpy()
        
        if out_list:
            o = out_list[i].data.cpu().numpy()
            # out = np.around(out[:, 1, ...])
            _, H, W = o.shape
            out = np.zeros((H, W))
            out[o[0] < o[1]] = 1
            
            plt.subplot(4, 3, i*3+1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                img = img * image_stats['std'] + image_stats['mean']
                plt.imshow(img)
            plt.subplot(4, 3, i*3+2)
            plt.imshow(gt, cmap='gray')
            plt.subplot(4, 3, i*3+3)
            plt.imshow(out, cmap='gray')
        else:
            plt.subplot(4, 2, i*2+1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.subplot(4, 2, i*2+2)
            plt.imshow(gt, cmap='gray')

    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def evaluate_error(out, target):
    errors = {'DIC': 0, 'JSC': 0}

    o = out.data.cpu().numpy()
    # out = np.around(out[:, 1, ...])
    N, _, H, W = o.shape
    out = np.zeros((N, H, W))
    out[o[:, 0] < o[:, 1]] = 1
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


def save_adversarial_imgs(x_adv, names, root):
    # print(x_adv.size())
    image_stats = {'mean':[0.7331, 0.6158, 0.5599],
                   'std':[0.1522, 0.1724, 0.1930]}
    
    for i in range(len(names)):
        img = x_adv[i].data.cpu().permute(1,2,0).squeeze().numpy()
        if len(img.shape) == 3:
            img = img * image_stats['std'] + image_stats['mean']
        img = np.clip(img, 0, 1)
        img = Image.fromarray(np.uint8(img*255))
        img = img.save(os.path.join(root, names[i]+'.png'))