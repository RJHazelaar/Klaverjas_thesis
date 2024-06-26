import numpy as np
import pandas as pd
import json
import os
import math
import tensorflow as tf

from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from Lennard.rounds import Round
from Lennard.deck import Card as Card_Lennard
from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.alphazero_player import AlphaZero_player
from AlphaZero.AlphaZeroPlayer.Klaverjas.helper import card_transform
from AlphaZero.AlphaZeroPlayer.networks import create_normal_nn

from AlphaZero.test_alphazero import run_test_multiprocess


def create_train_data(process_num: int, total_processes: int):

    data = pd.read_csv("Data/SL_Data/originalDB.csv", low_memory=False, converters={"Cards": pd.eval, "Rounds": eval})

    total_rounds = len(data.index)
    data_per_process = total_rounds // total_processes
    data = data[process_num * data_per_process : (process_num + 1) * data_per_process].reset_index(drop=True)

    X_train = np.zeros((data_per_process * 132, 299), dtype=np.float16)
    y_train = np.zeros((data_per_process * 132, 1), dtype=np.float16)
    mcts_params = {
        "mcts_steps": 10,
        "n_of_sims": 0,
        "nn_scaler": 0,
        "ucb_c": 50,
    }
    alpha_player_0 = AlphaZero_player(0, mcts_params, None)
    alpha_player_1 = AlphaZero_player(1, mcts_params, None)
    alpha_player_2 = AlphaZero_player(2, mcts_params, None)
    alpha_player_3 = AlphaZero_player(3, mcts_params, None)

    index = 0
    for round_num in range(data_per_process):
        scores = json.loads(data.loc[round_num]["Scores"])
        if scores["WeVerzaakt"] or scores["TheyVerzaakt"]:
            continue

        next_round = False
        if process_num == 0 and round_num % 100 == 0:
            print(round_num)

        round_obj = Round(
            data.loc[round_num]["FirstPlayer"], data.loc[round_num]["Troef"][0], data.loc[round_num]["Gaat"]
        )
        round_obj.set_cards(data.loc[round_num]["Cards"])
        alpha_player_0.new_round_Round(round_obj)
        alpha_player_1.new_round_Round(round_obj)
        alpha_player_2.new_round_Round(round_obj)
        alpha_player_3.new_round_Round(round_obj)

        round_score_we = scores["We"] + scores["WeRoem"]
        round_scores_they = scores["They"] + scores["TheyRoem"]

        # process the first 7 tricks
        for trick in data.loc[round_num]["Rounds"]:
            for _ in range(4):
                card = trick["Cards"][round_obj.current_player]
                card_object = Card_Lennard(int(card[1:]), card[0])
                # check if the card is legal
                if card_object not in round_obj.legal_moves(round_obj.current_player):
                    next_round = True
                    break

                X_train[index] = alpha_player_0.state.to_nparray()
                X_train[index + 1] = alpha_player_1.state.to_nparray()
                X_train[index + 2] = alpha_player_2.state.to_nparray()
                X_train[index + 3] = alpha_player_3.state.to_nparray()
                y_train[index] = round_score_we - round_scores_they
                y_train[index + 1] = round_scores_they - round_score_we
                y_train[index + 2] = round_score_we - round_scores_they
                y_train[index + 3] = round_scores_they - round_score_we
                round_obj.play_card(card_object)
                alpha_card = Card(card_transform(card_object.id, ["k", "h", "r", "s"].index(round_obj.trump_suit)))
                alpha_player_0.update_state(alpha_card)
                alpha_player_1.update_state(alpha_card)
                alpha_player_2.update_state(alpha_card)
                alpha_player_3.update_state(alpha_card)
                index += 4
            if next_round:
                break
        if next_round:
            continue

        # process the last trick
        for _ in range(4):
            X_train[index] = alpha_player_0.state.to_nparray()
            X_train[index + 1] = alpha_player_1.state.to_nparray()
            X_train[index + 2] = alpha_player_2.state.to_nparray()
            X_train[index + 3] = alpha_player_3.state.to_nparray()
            y_train[index] = round_score_we - round_scores_they
            y_train[index + 1] = round_scores_they - round_score_we
            y_train[index + 2] = round_score_we - round_scores_they
            y_train[index + 3] = round_scores_they - round_score_we
            index += 4

            card_object = round_obj.legal_moves(round_obj.current_player)[0]
            round_obj.play_card(card_object)
            alpha_card = Card(card_transform(card_object.id, ["k", "h", "r", "s"].index(round_obj.trump_suit)))
            alpha_player_0.update_state(alpha_card)
            alpha_player_1.update_state(alpha_card)
            alpha_player_2.update_state(alpha_card)
            alpha_player_3.update_state(alpha_card)
        X_train[index] = alpha_player_0.state.to_nparray()
        X_train[index + 1] = alpha_player_1.state.to_nparray()
        X_train[index + 2] = alpha_player_2.state.to_nparray()
        X_train[index + 3] = alpha_player_3.state.to_nparray()
        y_train[index] = round_score_we - round_scores_they
        y_train[index + 1] = round_scores_they - round_score_we
        y_train[index + 2] = round_score_we - round_scores_they
        y_train[index + 3] = round_scores_they - round_score_we

        if round_score_we - round_scores_they != int(round((X_train[index][-2] - X_train[index][-1]) * 100)):
            print(round_num)
            print(X_train[index][-2], X_train[index][-1])
            print(round_score_we - round_scores_they, X_train[index][-2] - X_train[index][-1])
            if scores["WeNat"] or scores["TheyNat"]:
                pass
            else:
                print("Something went wrong")
                print(process_num * data_per_process + round_num)

        index += 4
    X_train = X_train[:index]
    y_train = y_train[:index]

    train_data = np.concatenate((X_train, y_train), axis=1)

    np.save(f"Data/SL_Data/train_data_{process_num}.npy", train_data)
    # np.savetxt(f"Data/train_data_{process_num}.csv", train_data, delimiter=",")


def merge_npy(files):
    arrays = []
    for num in range(files):
        array = np.load(f"Data/SL_Data/train_data_{num}.npy")
        arrays.append(array)
    train_data = np.concatenate(arrays, axis=0)
    np.save(f"Data/SL_Data/train_data.npy", train_data)


def run_create_data():
    try:
        n_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        cluster = "cluster"
    except:
        n_cores = 10
        cluster = "local"
    print(cluster, "n_cores: ", n_cores)
    with Pool(processes=n_cores) as pool:
        pool.starmap(create_train_data, [(i, n_cores) for i in range(n_cores)])


def train_nn_on_data(model_name, step):

    print("loading data")
    data = np.load("Data/SL_Data/train_data.npy")
    print("data loaded")

    X = data[:, :299]
    y = data[:, 299]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, restore_best_weights=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = create_normal_nn(0.01, 0.0, 0.0)

    model.fit(
        X_train,
        y_train,
        batch_size=2048,
        epochs=5,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=early_stopping,
    )

    y_pred_test = model(X_test)

    print(X_test[:10][-5:])

    print(y_pred_test[:10])
    print(y_test[:10])

    model.save(f"Data/Models/SL_models/{model_name}_{step}.h5")


def test(test_rounds, mcts_params, model_name, step):
    scores_round, alpha_eval_time, _ = run_test_multiprocess(
        10, "rule", test_rounds, mcts_params, [f"SL_models/{model_name}_{step}.h5", None], True
    )

    mean_score = sum(scores_round) / len(scores_round)

    print(
        "score:",
        round(mean_score, 1),
        "std_score:",
        round(np.std(scores_round) / np.sqrt(len(scores_round)), 1),
        "eval_time(ms):",
        alpha_eval_time,
    )


def main():
    model_name = "SL_model"
    step = 1
    mcts_params = {
        "mcts_steps": 10,
        "n_of_sims": 0,
        "nn_scaler": 1,
        "ucb_c": 50,
    }
    test_rounds = 5000

    # run_create_data()
    # merge_npy(10)
    # train_nn_on_data(model_name, step)
    test(test_rounds, mcts_params, model_name, step)


if __name__ == "__main__":
    main()
