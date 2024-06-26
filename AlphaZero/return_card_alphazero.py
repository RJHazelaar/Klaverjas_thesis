from __future__ import annotations  # To use the class name in the type hinting

import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import sys

from multiprocessing import Pool, get_context
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../")))
sys.path.append(parent_dir)
data_dir = parent_dir+"/Data/SL_Data/TestAI-amsterdams.csv"

from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.helper import card_transform, card_untransform
from AlphaZero.AlphaZeroPlayer.alphazero_player import AlphaZero_player
from AlphaZero.AlphaZeroPlayer.alphazero_player_heavyrollout import AlphaZero_player_heavyrollout
from Lennard.rounds import Round
from Lennard.rule_based_agent import Rule_player

import csv



def play_cards_csv():
    rounds = pd.read_csv(data_dir, delimiter=",", low_memory=False)
    num_rounds = len(rounds)

    rounds["FirstPlayer"] = np.nan
    # TODO tijdelijke oplossing voor Gaat
    rounds["Gaat"] = 1

    print(rounds.head())

    # current_player
    

    return

    model1, model2 = None, None
    #if model_paths[0] is not None:
    #    model1 = tf.keras.models.load_model("Data/Models/" + model_paths[0])
    #if model_paths[1] is not None:
    #    model2 = tf.keras.models.load_model("Data/Models/" + model_paths[1])

    alpha_player_0 = AlphaZero_player(0, mcts_params, model1)

    for round_num in range(num_rounds):
        # if not process_id and round_num % 50 == 0:
        #     print(round_num)
        # round = Round((starting_player + 1) % 4, random.choice(['k', 'h', 'r', 's']), random.choice([0,1,2,3]))

        round = Round(
            rounds.loc[round_num]["FirstPlayer"], rounds.loc[round_num]["Troef"][0], rounds.loc[round_num]["Gaat"]
        )
        round.set_cards(rounds.loc[round_num]["Cards"])

        alpha_player_0.new_round_Round(round)

        for trick in range(8):
            for j in range(4):

                current_player = alpha_player_0.state.current_player

                tijd = time.time()
                if current_player == 0:
                    played_card = alpha_player_0.get_move()
                elif current_player == 1:
                    played_card = alpha_player_0.get_move()
                elif current_player == 2:
                    played_card = alpha_player_0.get_move()
                else:
                    played_card = alpha_player_0.get_move()
                alpha_eval_time += time.time() - tijd

                alpha_player_0.update_state(played_card)

    return

def run_test_multiprocess(
    n_cores: int, opponent: str, total_rounds: int, mcts_params: dict, model_paths: List[str], multiprocessing: bool
):

    test_function = 1
    rounds_per_process = total_rounds // n_cores
    if rounds_per_process == 0:
        raise Exception("too few rounds to test")
    if opponent == "rule":
        1 == 1
    elif opponent == "alphazero":
        1 == 1
    elif opponent == "rule_heavy":
        1 == 1
    else:
        raise Exception("mode not found")

    scores_round = []
    points_cumulative = [0, 0]
    mcts_times = [0, 0, 0, 0, 0]
    alpha_eval_time = 0
    if multiprocessing:
        with get_context("spawn").Pool(processes=n_cores) as pool:
            items = [(rounds_per_process, i, mcts_params, model_paths) for i in range(n_cores)]
            for result in pool.starmap(test_function, items):
                scores_round += result[0]
                points_cumulative[0] += result[1][0]
                points_cumulative[1] += result[1][1]
                mcts_times = [mcts_times[i] + result[2][i] for i in range(len(mcts_times))]
                alpha_eval_time += result[3]

        if len(scores_round) != n_cores * rounds_per_process:
            print(len(scores_round))
            print(scores_round)
            raise Exception("wrong length")
    else:
        scores_round, points_cumulative, mcts_times, alpha_eval_time = test_function(
            rounds_per_process, 0, mcts_params, model_paths
        )

    alpha_eval_time /= total_rounds * 8
    alpha_eval_time = round(alpha_eval_time * 1000, 1)  # convert to ms and round to 1 decimal
    temp = None
    return scores_round, alpha_eval_time, temp

if __name__ == "__main__":
    play_cards_csv()
    print("All cards were played.")