import numpy as np
import time
import os
import math
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../..")))
sys.path.append(parent_dir)
experiment_dir = parent_dir + "/Experiments/versus_rule_based"
sys.path.append(experiment_dir)
from AlphaZero.experiment_alphazero import run_test_multiprocess
import run_experiment_alphazero_1
import run_experiment_alphazero_2
import run_experiment_alphazero_3
import run_experiment_alphazero_4
import run_experiment_alphazero_5
import run_experiment_heavy_10
import run_experiment_heavy_6
import run_experiment_heavy_7
import run_experiment_heavy_8
import run_experiment_heavy_9


def run_experiments():
    run_experiment_alphazero_1.run_test()
    run_experiment_alphazero_2.run_test()
    run_experiment_alphazero_3.run_test()
    run_experiment_alphazero_4.run_test()
    #run_experiment_alphazero_5.run_test()
    #run_experiment_heavy_10.run_test()
    #run_experiment_heavy_6.run_test()
    #run_experiment_heavy_7.run_test()
    #run_experiment_heavy_8.run_test()
    #run_experiment_heavy_9.run_test()

if __name__ == "__main__":
    start_time = time.time()
    run_experiments()
    print("Total time: ", time.time() - start_time)
