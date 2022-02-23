import time
import sys
import os

def message(config, string, time_stamp=True):
    t = time.localtime()
    if time_stamp:
        prefix = f"[{str(t.tm_hour).zfill(2)}:{str(t.tm_min).zfill(2)}:{str(t.tm_sec).zfill(2)}] "
    else:
        prefix = " "*11

    message = prefix + string + '\n'

    _print(config, message)
    
def line(config):
    _print(config, "-------------------------------------\n")

def config_init(config):
    if config.out_to_folder == "True":
        t = int(time.time())
        if 'results' not in os.listdir():
            os.mkdir('results')
        results_dir = os.path.join('results'+str(t))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        config.results_dir = results_dir

    _print(config, config)
    _print(config, "\n")
    line(config)

    return config

def _print(config, message):
    if config.out_to_folder == "True":
        with open(config.results_dir+"/out.debug.log", "a+") as f:
            sys.stdout = f
            print(message, end='')
            sys.stdout = sys.stdout

    print(message, end='')