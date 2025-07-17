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
import run_experiment_alphazero_cheat_1_e1
import run_experiment_alphazero_cheat_1_e025
import run_experiment_alphazero_cheat_1_e050
import run_experiment_alphazero_cheat_1_e075
import run_experiment_alphazero_cheat_2_e1
import run_experiment_alphazero_cheat_2_e025
import run_experiment_alphazero_cheat_2_e050
import run_experiment_alphazero_cheat_2_e075
import run_experiment_alphazero_cheat_3_e1
import run_experiment_alphazero_cheat_3_e025
import run_experiment_alphazero_cheat_3_e050
import run_experiment_alphazero_cheat_3_e075
import run_experiment_alphazero_cheat_4_e1
import run_experiment_alphazero_cheat_4_e025
import run_experiment_alphazero_cheat_4_e050
import run_experiment_alphazero_cheat_4_e075
import run_experiment_alphazero_cheat_5_e1
import run_experiment_alphazero_cheat_5_e025
import run_experiment_alphazero_cheat_5_e050
import run_experiment_alphazero_cheat_5_e075

def run_experiments():
    run_experiment_alphazero_cheat_1_e025.run_test()
    run_experiment_alphazero_cheat_1_e050.run_test()
    run_experiment_alphazero_cheat_1_e075.run_test()
    run_experiment_alphazero_cheat_1_e1.run_test()
    run_experiment_alphazero_cheat_2_e025.run_test()
    run_experiment_alphazero_cheat_2_e050.run_test()
    run_experiment_alphazero_cheat_2_e075.run_test()
    run_experiment_alphazero_cheat_2_e1.run_test()
    run_experiment_alphazero_cheat_3_e025.run_test()
    run_experiment_alphazero_cheat_3_e050.run_test()
    run_experiment_alphazero_cheat_3_e075.run_test()
    run_experiment_alphazero_cheat_3_e1.run_test()
    run_experiment_alphazero_cheat_4_e025.run_test()
    run_experiment_alphazero_cheat_4_e050.run_test()
    run_experiment_alphazero_cheat_4_e075.run_test()
    run_experiment_alphazero_cheat_4_e1.run_test()
    run_experiment_alphazero_cheat_5_e025.run_test()
    run_experiment_alphazero_cheat_5_e050.run_test()
    run_experiment_alphazero_cheat_5_e075.run_test()
    run_experiment_alphazero_cheat_5_e1.run_test()

if __name__ == "__main__":
    start_time = time.time()
    run_experiments()
    print("Total time: ", time.time() - start_time)
