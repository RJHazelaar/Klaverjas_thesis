import numpy as np
import time
import os
import math
import sys

parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../..")))
sys.path.append(parent_dir)
from AlphaZero.experiment_heavy_pimc import run_test_multiprocess


def run_test():
    try:
        n_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        cluster = "cluster"
    except:
        n_cores = 10
        cluster = "local"

    #TODO Klopt dit????
    n_cores = os.cpu_count()
    n_cores = 20
    #TODO rule for heavy vs rule
    #TODO alphazero for heavy vs standard_alphazero
    # rule, rule_heavy, alphazero
    # Alphazero for heavy vs random rollout test
    opponent = "pimc"
    multiprocessing = True #TODO

    num_rounds = 10000
    num_rounds = (
        math.ceil(num_rounds / n_cores) * n_cores
    )  # make sure rounds is divisible by n_cores and not devide to 0

    mcts_params = {
        "mcts_steps": 10,
        "n_of_sims": 1,
        "nn_scaler": 0,
        "ucb_c": 200,
        "steps_per_determinization": 10,
    }

    # model_paths = ["SL_models/SL_model_0.h5", None]
    model_paths = [None, None]

    print(
        "cluster:",
        cluster,
        "cores",
        n_cores,
        "rounds:",
        num_rounds,
        "mcts_params:",
        mcts_params,
        "model_paths:",
        model_paths,
    )
    now = time.time()
    scores_round, alpha_eval_time, _ = run_test_multiprocess(
        n_cores, opponent, num_rounds, mcts_params, model_paths, multiprocessing
    )
    print("results exp6", mcts_params)
    print("time:", time.time() - now)
    mean_score = sum(scores_round) / len(scores_round)

    odd_i = []
    even_i = []

    for i in range(0, len(scores_round)):
        if i % 2:
            even_i.append(scores_round[i])
        else :
            odd_i.append(scores_round[i])

    scores_round_alt = []
    for l1, l2 in zip(odd_i, even_i):
         scores_round_alt.append(2)
         
    print(
        "score:",
        round(mean_score, 1),
        "std_score:",
        round(np.std(scores_round), 1),
        "std_error",
        round(np.std(scores_round) / np.sqrt(len(scores_round)), 1),
        "eval_time(ms):",
        alpha_eval_time,
    )


if __name__ == "__main__":
    start_time = time.time()
    run_test()
    print("Total time: ", time.time() - start_time)
    print("Experiments/versus_rule_based_ensemble/run_experiment_heavy_pimc_1.py")