import os
import wandb
import math
import time
import sys

from AlphaZero.train_alphazero_master_split import train
from AlphaZero.AlphaZeroPlayer.networks import create_simple_nn, create_normal_two_headed_nn, create_bidding_nn, create_bidding_nn_alt, create_value_nn, create_policy_nn

parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../")))
data_dir = "/local/s1762508"


def run_train(
    run_settings,
    model_params,
    bidding_model_params,
    selfplay_params,
    fit_params,
    test_params,
):
    budget = run_settings["budget"]
    n_cores = run_settings["n_cores"]
    starting_step = run_settings["starting_step"]
    model_value_name = run_settings["model_value_name"]
    model_policy_name = run_settings["model_policy_name"]
    bidding_model_name = run_settings["bidding_model_name"]
    multiprocessing = run_settings["multiprocessing"]
    learning_rate = model_params["learning_rate"]
    l1 = model_params["l1"]
    l2 = model_params["l2"]
    learning_rate_bidding = bidding_model_params["learning_rate"]
    l1_bidding = bidding_model_params["l1"]
    l2_bidding = bidding_model_params["l2"]
    rounds_per_step = selfplay_params["rounds_per_step"]
    extra_noise_ratio = selfplay_params["extra_noise_ratio"]
    training_size_multiplier = fit_params["training_size_multiplier"]
    mcts_params = selfplay_params["mcts_params"]
    max_memory_multiplier = selfplay_params["max_memory_multiplier"]
    jumpstart =  run_settings["jumpstart"]

    test_params["test_rounds"] = (
        math.ceil(test_params["test_rounds"] / n_cores) * n_cores
    )  # make sure rounds is divisible by n_cores and not devide to 0
    rounds_per_step = (
        math.ceil(rounds_per_step / n_cores) * n_cores
    )  # make sure rounds is divisible by n_cores and not devide to 0
    selfplay_params["rounds_per_step"] = rounds_per_step
    max_memory = rounds_per_step * 36 * max_memory_multiplier

    wandb.init(
        # set the wandb project where this run will be logged
        project=run_settings["project_name"],
        # name of the run
        name=model_value_name,
        # track hyperparameters and run metadata
        config={
            "run_settings": run_settings,
            "model_params": model_params,
            "selfplay_params": selfplay_params,
            "fit_params": fit_params,
            "test_params": test_params,
        },
    )
    print("starting training")
    print("run settings:", run_settings)
    print("model params:", model_params)
    print("bidding model params:", bidding_model_params)
    print("selfplay params:", selfplay_params)
    print("fit params:", fit_params)
    print("test params:", test_params)

    if starting_step == 0:
        print(f"{data_dir}/Klaverjas_thesis-main/Data/RL_data/{model_value_name}/")
        #print(f"{parent_dir}/Klaverjas_thesis-main/Data/RL_data/{model_name}/")
        try:
            os.mkdir(f"{data_dir}/Klaverjas_thesis-main/Data/RL_data/{model_value_name}/")
            #os.mkdir(f"{parent_dir}/Klaverjas_thesis-main/Data/RL_data/{model_name}/")
        except:
            print("\n\n\n============model already exists============\n\n\n")
        if model_params["model_type"] == "simple":
            model = create_simple_nn(learning_rate, l1, l2)
        elif model_params["model_type"] == "two_headed":
            model = create_normal_two_headed_nn(learning_rate, l1, l2)
        elif model_params["model_type"] == "split":
            model_value = create_value_nn(learning_rate, l1, l2)
            model_policy = create_policy_nn(learning_rate, l1, l2)
        else:
            raise Exception("model type not recognized")

        model_value.save(f"{data_dir}/Klaverjas_thesis-main/Data/Models/{model_value_name}/{model_value_name}_0.h5")
        model_policy.save(f"{data_dir}/Klaverjas_thesis-main/Data/Models/{model_value_name}/{model_policy_name}_0.h5")
        #model.save(f"{parent_dir}/Klaverjas_thesis-main/Data/Models/{model_name}/{model_name}_0.h5")
#######################################################################################################
        print(f"{data_dir}/Klaverjas_thesis-main/Data/RL_data/{bidding_model_name}/")
        try:
            os.mkdir(f"{data_dir}/Klaverjas_thesis-main/Data/RL_data/{bidding_model_name}/")
            #os.mkdir(f"{parent_dir}/Klaverjas_thesis-main/Data/RL_data/{bidding_model_name}/")
        except:
            print("\n\n\n============bidding model already exists============\n\n\n")
        if bidding_model_params["model_type"] == "bidding":
            bidding_model = create_bidding_nn(learning_rate_bidding, l1_bidding, l2_bidding)
        elif bidding_model_params["model_type"] == "bidding_alt":
            bidding_model = create_bidding_nn_alt(learning_rate_bidding, l1_bidding, l2_bidding)
        else:
            raise Exception("model type not recognized")

        bidding_model.save(f"{data_dir}/Klaverjas_thesis-main/Data/Models/{bidding_model_name}/{bidding_model_name}_0.h5")
        #bidding_model.save(f"{parent_dir}/Klaverjas_thesis-main/Data/Models/{bidding_model_name}/{bidding_model_name}_0.h5")

    total_time, selfplay_time, training_time, testing_time = train(
        budget,
        starting_step,
        model_value_name,
        model_policy_name,
        bidding_model_name,
        max_memory,
        multiprocessing,
        n_cores,
        rounds_per_step,
        training_size_multiplier,
        mcts_params,
        fit_params,
        test_params,
        extra_noise_ratio,
        jumpstart,
    )
    print("total time:", total_time)
    print("selfplay time:", selfplay_time)
    print("training time:", training_time)
    print("testing time:", testing_time)
    wandb.finish()


def main():
    try:
        n_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        cluster = "cluster"
    except:
        n_cores = 20
        cluster = "local"
    print(f"Using {n_cores} cores on {cluster}")

    model_value_name = "value_network"
    model_policy_name = "policy_network"
    bidding_model_name = "bidding_network_policy"
    run_settings = {
        "project_name": "Thesis_test_policy",
        "model_value_name": model_value_name,
        "model_policy_name": model_policy_name,
        "bidding_model_name": bidding_model_name,
        "starting_step": 0,
        "budget": 12,  # hours
        "multiprocessing": True, #TODO
        "n_cores": n_cores,
        "jumpstart": 0,
    }
    model_params = {
        "model_type": "split",
        "learning_rate": 0.01,
        "l1": 0.0001,
        "l2": 0.0001,
    }
    bidding_model_params = {
        "model_type": "bidding_alt",
        "learning_rate": 0.01,
        "l1": 0.0001,
        "l2": 0.0001,
    }
    selfplay_params = {
        "rounds_per_step": 40,  # amount of selfplay rounds per step
        "max_memory_multiplier": 10,  # memory size = rounds_per_step * 36 * max_memory_multiplier
        "extra_noise_ratio": 0.1,  # when training extra_noise_ratio * mcts_steps is added to all visit counts
        "mcts_params": {
            "mcts_steps": 200,
            "n_of_sims": 1,
            "nn_scaler": 0,
            "ucb_c": 1.4,
            "steps_per_determinization": 40,
        },
    }
    fit_params = {
        "training_size_multiplier": 3,  # training size = training_size_multiplier * rounds_per_step * 36
        "epochs": 1,
        "batch_size": 2048,
    }
    test_params = {
        "test_rounds": 10000,
        "test_frequency":1,
        "mcts_params": {
            "mcts_steps": 10,
            "n_of_sims": 0,
            "nn_scaler": 1,
            "ucb_c": 1.4,
            "steps_per_determinization": 10,
        },
    }
    run_train(
        run_settings,
        model_params,
        bidding_model_params,
        selfplay_params,
        fit_params,
        test_params,
    )


if __name__ == "__main__":
    starting_time = time.time()
    main()
    print("total run time:", time.time() - starting_time)
