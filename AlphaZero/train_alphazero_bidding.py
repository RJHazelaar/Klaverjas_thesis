from __future__ import annotations  # To use the class name in the type hinting

import os
import numpy as np
import random
import time
import tensorflow as tf
import wandb

from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from multiprocessing import get_context

from AlphaZero.AlphaZeroPlayer.alphazero_player import AlphaZero_player
from AlphaZero.test_alphazero import run_test_multiprocess
from Lennard.rounds import Round

parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../")))
data_dir = "/local/s1762508/Klaverjas_thesis-main"


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


def selfplay(mcts_params, model_path, bidding_model_path, num_rounds, extra_noise_ratio):
    if model_path is not None:
        model = tf.keras.models.load_model(f"{data_dir}/Data/Models/{model_path}")
        #model = tf.keras.models.load_model(f"{parent_dir}/Data/Models/{model_path}")
    else:
        model = None

    if bidding_model_path is not None:
        bidding_model = tf.keras.models.load_model(f"{data_dir}/Data/Models/{bidding_model_path}")
        #bidding_model = tf.keras.models.load_model(f"{parent_dir}/Data/Models/{bidding_model_path}")
    else:
        bidding_model = None

    # 32 turns + 4 end states, 1 for each player
    X_train = np.zeros((num_rounds * 36, 299), dtype=np.float16)
    y_train = np.zeros((num_rounds * 36, 1), dtype=np.float16)

    # 4 player that can bid, 32 cards + 4 one hot encoded player
    X_train_bid = np.zeros((num_rounds * 4, 36), dtype=np.float16)
    y_train_bid = np.zeros((num_rounds * 4, 5), dtype=np.float16)

    alpha_player_0 = AlphaZero_player(0, mcts_params, model)
    alpha_player_1 = AlphaZero_player(1, mcts_params, model)
    alpha_player_2 = AlphaZero_player(2, mcts_params, model)
    alpha_player_3 = AlphaZero_player(3, mcts_params, model)

    for round_num in range(num_rounds):
        starting_player = declarer = random.choice([0, 1, 2, 3])
        round = Round(starting_player, random.choice(["k", "h", "r", "s"]), declarer, bidding_model)
        
        X_train_bid[round_num * 4] = round.hand_to_input_vector(0, starting_player)
        X_train_bid[round_num * 4 + 1] = round.hand_to_input_vector(1, starting_player)
        X_train_bid[round_num * 4 + 2] = round.hand_to_input_vector(2, starting_player)
        X_train_bid[round_num * 4 + 3] = round.hand_to_input_vector(3, starting_player)

        alpha_player_0.new_round_Round(round)
        alpha_player_1.new_round_Round(round)
        alpha_player_2.new_round_Round(round)
        alpha_player_3.new_round_Round(round)

        # generate a state and score and play a card
        for trick in range(8):
            for _ in range(4):
                current_player = alpha_player_0.state.current_player

                if current_player == 0:
                    played_card = alpha_player_0.get_move(True, extra_noise_ratio)
                    X_train[round_num * 36 + trick * 4] = alpha_player_0.state.to_nparray()
                elif current_player == 1:
                    played_card = alpha_player_1.get_move(True, extra_noise_ratio)
                    X_train[round_num * 36 + trick * 4 + 1] = alpha_player_1.state.to_nparray()
                elif current_player == 2:
                    played_card = alpha_player_2.get_move(True, extra_noise_ratio)
                    X_train[round_num * 36 + trick * 4 + 2] = alpha_player_2.state.to_nparray()
                else:
                    played_card = alpha_player_3.get_move(True, extra_noise_ratio)
                    X_train[round_num * 36 + trick * 4 + 3] = alpha_player_3.state.to_nparray()

                alpha_player_0.update_state(played_card)
                alpha_player_1.update_state(played_card)
                alpha_player_2.update_state(played_card)
                alpha_player_3.update_state(played_card)

        # generate state and score for end state
        X_train[round_num * 36 + 32] = alpha_player_0.state.to_nparray()
        X_train[round_num * 36 + 32 + 1] = alpha_player_1.state.to_nparray()
        X_train[round_num * 36 + 32 + 2] = alpha_player_2.state.to_nparray()
        X_train[round_num * 36 + 32 + 3] = alpha_player_3.state.to_nparray()

        score_player_0 = alpha_player_0.state.get_score(0)
        score_player_1 = alpha_player_1.state.get_score(1)
        score_player_2 = alpha_player_2.state.get_score(2)
        score_player_3 = alpha_player_3.state.get_score(3)

        if score_player_0 != score_player_2 or score_player_1 != score_player_3:
            raise Exception("Scores are not equal")
        if score_player_0 + score_player_1 + score_player_2 + score_player_3 != 0:
            raise Exception("Scores do not add up to 0")

        for trick in range(9):
            y_train[round_num * 36 + trick * 4] = score_player_0
            y_train[round_num * 36 + trick * 4 + 1] = score_player_1
            y_train[round_num * 36 + trick * 4 + 2] = score_player_2
            y_train[round_num * 36 + trick * 4 + 3] = score_player_3

        for bidder, player in zip(round.bidders, [0,1,2,3]):
            X_train_bid[round_num * 4 + player] = X_train_bid[round_num * 4 + player] * bidder
            y_train_bid[round_num * 4 + player] = alpha_player_0.state.get_prediction_score(0, round.declarer, round.trump_suit) * bidder


    train_data = np.concatenate((X_train, y_train), axis=1)

    # First remove training data for players that didnt bid
    X_train_bid = X_train_bid[~np.all(X_train_bid == 0, axis=1)]
    y_train_bid = y_train_bid[~np.all(y_train_bid == 0, axis=1)]
    train_data_bidding = np.concatenate((X_train_bid, y_train_bid), axis=1)
    return train_data, train_data_bidding


def train_nn(train_data, model: tf.keras.Sequential, fit_params, callbacks):
    epochs = fit_params["epochs"]
    batch_size = fit_params["batch_size"]

    X_train, X_test, y_train, y_test = train_test_split(
        train_data[:, :299], train_data[:, 299], train_size=0.8, shuffle=True
    )

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

def train_bidding_nn(train_data, model: tf.keras.Sequential, fit_params, callbacks):
    epochs = fit_params["epochs"]
    batch_size = fit_params["batch_size"]

    X_train, X_test, y_train, y_test = train_test_split(
        train_data[:, :36], train_data[:, 36], train_size=0.8, shuffle=True
    )

    y_train = to_categorical(y_train, 5)
    y_test = to_categorical(y_test, 5)

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

def train(
    budget,
    step,
    model_name,
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
):
    start_time = time.time()
    total_selfplay_time = 0
    total_training_time = 0
    total_testing_time = 0

    # budget in seconds
    budget = budget * 3600
    test_rounds = test_params["test_rounds"]
    test_frequency = test_params["test_frequency"]
    test_mcts_params = test_params["mcts_params"]

    if step == 0:
        memory = None
        bidding_memory = None
    else:
        memory = np.load(f"{data_dir}/Data/RL_data/{model_name}/{model_name}_{step}_memory.npy")
        #memory = np.load(f"{parent_dir}/Data/RL_data/{model_name}/{model_name}_{step}_memory.npy")
        bidding_memory = np.load(f"{data_dir}/Data/RL_data/{bidding_model_name}/{bidding_model_name}_{step}_memory.npy")
        #bidding_memory = np.load(f"{parent_dir}/Data/RL_data/{bidding_model_name}/{bidding_model_name}_{step}_memory.npy")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=0, restore_best_weights=True)
    wandb.log({"Average Score": -35, "Train Time": 0})
    model_path = f"{model_name}/{model_name}_{step}.h5"
    bidding_model_path = f"{bidding_model_name}/{bidding_model_name}_{step}.h5"

    while time.time() - start_time < budget:
        step += 1

        # generate data
        tijd = time.time()
        if multiprocessing:
            with get_context("spawn").Pool(processes=n_cores) as pool:
                data, bidding_data = zip(*pool.starmap(
                    selfplay,
                    [(mcts_params, model_path, bidding_model_path, rounds_per_step // n_cores, extra_noise_ratio) for _ in range(n_cores)],
                ))
            data = np.concatenate(data, axis=0)
            bidding_data = np.concatenate(bidding_data, axis=0)
        else:
            data, bidding_data = selfplay(mcts_params, model_path, bidding_model_path, rounds_per_step, extra_noise_ratio)
        selfplay_time = time.time() - tijd

        np.save(f"{data_dir}/Data/RL_data/{model_name}/{model_name}_{step}.npy", data)
        #np.save(f"{parent_dir}/Data/RL_data/{model_name}/{model_name}_{step}.npy", data)
        np.save(f"{data_dir}/Data/RL_data/{bidding_model_name}/{bidding_model_name}_{step}.npy", bidding_data)
        #np.save(f"{parent_dir}/Data/RL_data/{bidding_model_name}/{bidding_model_name}_{step}.npy", bidding_data)

        # add data to memory and remove old data if memory is full
        if memory is None:
            memory = data
        else:
            memory = np.concatenate((memory, data), axis=0)
        if len(memory) > max_memory:
            memory = np.delete(memory, np.s_[0 : len(memory) - max_memory], axis=0)

        if bidding_memory is None:
            bidding_memory = bidding_data
        else:
            bidding_memory = np.concatenate((bidding_memory, bidding_data), axis=0)
        if len(bidding_memory) > max_memory:
            bidding_memory = np.delete(bidding_memory, np.s_[0 : len(bidding_memory) - max_memory], axis=0)

        # select train data and train model
        train_data = memory[
            np.random.choice(len(memory), rounds_per_step * 36 * training_size_multiplier, replace=False), :
        ]

        # load train and save model
        model = tf.keras.models.load_model(f"{data_dir}/Data/Models/{model_path}")
        #model = tf.keras.models.load_model(f"{parent_dir}/Data/Models/{model_path}")
        tijd = time.time()
        train_nn(train_data, model, fit_params, [early_stopping])
        training_time = time.time() - tijd
        model_path = f"{model_name}/{model_name}_{step}.h5"
        if step == 240:
            tf.keras.backend.set_value(
                model.optimizer.learning_rate,
                tf.keras.backend.get_value(model.optimizer.learning_rate) / 10,
            )
        model.save(f"{data_dir}/Data/Models/{model_path}")
        #model.save(f"{parent_dir}/Data/Models/{model_path}")

        # Same but for bidding network

        # select train data and train model
        if (rounds_per_step * 4 * training_size_multiplier < len(bidding_memory)):
            bidding_train_data = bidding_memory[
                np.random.choice(len(bidding_memory), rounds_per_step * 4 * training_size_multiplier, replace=False), :
            ]
        else:
            bidding_train_data = bidding_memory[
                np.random.choice(len(bidding_memory), len(bidding_memory), replace=False), :
            ]
        # load train and save bidding model
        bidding_model = tf.keras.models.load_model(f"{data_dir}/Data/Models/{bidding_model_path}")
        #bidding_model = tf.keras.models.load_model(f"{parent_dir}/Data/Models/{bidding_model_path}")
        tijd = time.time()
        train_bidding_nn(bidding_train_data, bidding_model, fit_params, [early_stopping])
        training_time = time.time() - tijd
        bidding_model_path = f"{bidding_model_name}/{bidding_model_name}_{step}.h5"
        if step == 240:
            tf.keras.backend.set_value(
                bidding_model.optimizer.learning_rate,
                tf.keras.backend.get_value(bidding_model.optimizer.learning_rate) / 10,
            )
        bidding_model.save(f"{data_dir}/Data/Models/{bidding_model_path}")
        #bidding_model.save(f"{parent_dir}/Data/Models/{bidding_model_path}")

        total_selfplay_time += selfplay_time
        total_training_time += training_time

        tijd = time.time()
        if step % test_frequency == 0:
            scores_round, _, _ = run_test_multiprocess(
                n_cores, "rule", test_rounds, test_mcts_params, [model_path, None], multiprocessing
            )
            wandb.log(
                {
                    "Average Score": sum(scores_round) / len(scores_round),
                    "Train Time": total_selfplay_time + total_training_time,
                }
            )
        total_testing_time += time.time() - tijd

    # always test at the end
    if step % test_frequency != 0:
        scores_round, _, _ = run_test_multiprocess(
            n_cores, "rule", test_rounds, test_mcts_params, [model_path, None], multiprocessing
        )
        wandb.log(
            {
                "Average Score": sum(scores_round) / len(scores_round),
                "Train Time": total_selfplay_time + total_training_time,
            }
        )
    np.save(f"{data_dir}/Data/RL_data/{model_name}/{model_name}_{step}_memory.npy", memory)
    #np.save(f"{parent_dir}/Data/RL_data/{model_name}/{model_name}_{step}_memory.npy", memory)
    return time.time() - start_time, total_selfplay_time, total_training_time, total_testing_time
