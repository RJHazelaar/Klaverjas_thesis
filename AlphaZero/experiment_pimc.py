from __future__ import annotations  # To use the class name in the type hinting

import os
import time
import tensorflow as tf
import pandas as pd
import sys

from multiprocessing import Pool, get_context
from typing import List

from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.helper import card_transform, card_untransform
from AlphaZero.AlphaZeroPlayer.alphazero_player import AlphaZero_player
from AlphaZero.AlphaZeroPlayer.pimc_player import PIMC_player
from Lennard.rounds import Round
from Lennard.rule_based_agent import Rule_player

import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../")))
sys.path.append(parent_dir)
data_dir = parent_dir+"/Data/SL_Data/originalDB.csv"

def test_vs_alphazero_player(
    num_rounds: int,
    process_id: int,
    mcts_params: dict,
    model_paths: List[str],
):
    mcts_times = [0, 0, 0, 0, 0]
    alpha_eval_time = 0
    point_cumulative = [0, 0]
    scores_alpha = []

    if num_rounds * (process_id + 1) > 50010:
        raise "too many rounds"

    rounds = pd.read_csv(data_dir, delimiter=";", low_memory=False, converters={"Cards": pd.eval})

    model1, model2 = None, None
    if model_paths[0] is not None:
        model1 = tf.keras.models.load_model("Data/Models/" + model_paths[0])
    if model_paths[1] is not None:
        model2 = tf.keras.models.load_model("Data/Models/" + model_paths[1])

    alpha_player_0 = AlphaZero_player(0, mcts_params, model1)
    alpha_player_1 = AlphaZero_player(1, mcts_params, model2)
    alpha_player_2 = AlphaZero_player(2, mcts_params, model1)
    alpha_player_3 = AlphaZero_player(3, mcts_params, model2)

    for round_num in range(num_rounds * process_id, num_rounds * (process_id + 1)):
        # if not process_id and round_num % 50 == 0:
        #     print(round_num)
        # round = Round((starting_player + 1) % 4, random.choice(['k', 'h', 'r', 's']), random.choice([0,1,2,3]))

        round = Round(
            rounds.loc[round_num]["FirstPlayer"], rounds.loc[round_num]["Troef"][0], rounds.loc[round_num]["Gaat"]
        )
        round.set_cards(rounds.loc[round_num]["Cards"])

        alpha_player_0.new_round_Round(round)
        alpha_player_1.new_round_Round(round)
        alpha_player_2.new_round_Round(round)
        alpha_player_3.new_round_Round(round)

        for trick in range(8):
            for j in range(4):

                current_player = alpha_player_0.state.current_player

                tijd = time.time()
                if current_player == 0:
                    played_card = alpha_player_0.get_move()
                elif current_player == 1:
                    played_card = alpha_player_1.get_move()
                elif current_player == 2:
                    played_card = alpha_player_2.get_move()
                else:
                    played_card = alpha_player_3.get_move()
                alpha_eval_time += time.time() - tijd

                alpha_player_0.update_state(played_card)
                alpha_player_1.update_state(played_card)
                alpha_player_2.update_state(played_card)
                alpha_player_3.update_state(played_card)

        for i in range(5):
            mcts_times[i] += alpha_player_0.tijden[i]

        scores_alpha.append(alpha_player_0.state.get_score(0))

        point_cumulative[0] += round.points[0] + round.meld[0]
        point_cumulative[1] += round.points[1] + round.meld[1]

    return scores_alpha, point_cumulative, mcts_times, alpha_eval_time / 4


def test_vs_rule_player_pimc(
    num_rounds: int,
    process_id: int,
    mcts_params: dict,
    model_paths: List[str],
):
    # random.seed(13)
    alpha_eval_time = 0
    mcts_times = [0, 0, 0, 0, 0]
    point_cumulative = [0, 0]
    scores_alpha = []
    scores_round = []

    if num_rounds * (process_id + 1) > 50010:
        raise "too many rounds"

    rounds = pd.read_csv(data_dir, delimiter=";", low_memory=False, converters={"Cards": pd.eval})

    rule_player = Rule_player()

    model = None
    if model_paths[0] is not None:
        try:
            model = tf.keras.models.load_model("Data/Models/" + model_paths[0])
        except:
            print("model not found")
            raise Exception("model not found")

    alpha_player_0 = PIMC_player(0, mcts_params, model)
    alpha_player_2 = PIMC_player(2, mcts_params, model)

    for round_num in range(num_rounds * process_id, num_rounds * (process_id + 1)):
        # round = Round((starting_player + 1) % 4, random.choice(['k', 'h', 'r', 's']), random.choice([0,1,2,3]))

        if round_num % 2 == 0:
            round = Round(
                rounds.loc[round_num]["FirstPlayer"], rounds.loc[round_num]["Troef"][0], rounds.loc[round_num]["Gaat"]
            )
            round.set_cards(rounds.loc[round_num]["Cards"])
        else:
            round = Round(
                (rounds.loc[round_num - 1]["FirstPlayer"] + 1) % 4,
                rounds.loc[round_num - 1]["Troef"][0],
                (rounds.loc[round_num - 1]["Gaat"] + 1) % 4,
            )
            cards = rounds.loc[round_num - 1]["Cards"]
            round.set_cards(cards[3:] + cards[:3])

        alpha_player_0.new_round_Round(round)
        alpha_player_2.new_round_Round(round)
        for trick in range(8):
            for j in range(4):

                current_player = round.current_player
                if current_player == 1 or current_player == 3:

                    played_card = rule_player.get_card_good_player(round, current_player)
                    # moves = round.legal_moves()
                    # played_card = random.choice(moves)
                else:
                    tijd = time.time()
                    if current_player == 0:
                        played_card = alpha_player_0.get_move()
                        played_card = card_untransform(played_card.id, ["k", "h", "r", "s"].index(round.trump_suit))
                    else:
                        played_card = alpha_player_2.get_move()
                        played_card = card_untransform(played_card.id, ["k", "h", "r", "s"].index(round.trump_suit))
                    alpha_eval_time += time.time() - tijd
                    moves = round.legal_moves()

                    found = False
                    for move in moves:
                        if move.id == played_card:
                            played_card = move
                            found = True
                            break
                    if not found:
                        raise Exception("move not found")

                    # moves = round.legal_moves()
                    # played_card = random.choice(moves)

                round.play_card(played_card)
                move = Card(card_transform(played_card.id, ["k", "h", "r", "s"].index(round.trump_suit)))
                alpha_player_0.update_state(move)
                alpha_player_2.update_state(move)

        for i in range(5):
            mcts_times[i] += alpha_player_0.tijden[i]

        scores_alpha.append(alpha_player_0.state.get_score(0))
        scores_round.append(round.get_score(0))
        if scores_alpha[-1] != scores_round[-1]:
            print("scores_alpha not always equal to scores_round")
            print(scores_alpha[-1], scores_round[-1])
        point_cumulative[0] += round.points[0] + round.meld[0]
        point_cumulative[1] += round.points[1] + round.meld[1]

    return scores_round, point_cumulative, mcts_times, alpha_eval_time / 2

def test_vs_rule_player(
    num_rounds: int,
    process_id: int,
    mcts_params: dict,
    model_paths: List[str],
):
    # random.seed(13)
    alpha_eval_time = 0
    mcts_times = [0, 0, 0, 0, 0]
    point_cumulative = [0, 0]
    scores_alpha = []
    scores_round = []

    if num_rounds * (process_id + 1) > 50010:
        raise "too many rounds"

    rounds = pd.read_csv(data_dir, delimiter=";", low_memory=False, converters={"Cards": pd.eval})

    rule_player = Rule_player()

    model = None
    if model_paths[0] is not None:
        try:
            model = tf.keras.models.load_model("Data/Models/" + model_paths[0])
        except:
            print("model not found")
            raise Exception("model not found")

    alpha_player_0 = AlphaZero_player(0, mcts_params, model)
    alpha_player_2 = AlphaZero_player(2, mcts_params, model)

    for round_num in range(num_rounds * process_id, num_rounds * (process_id + 1)):
        # round = Round((starting_player + 1) % 4, random.choice(['k', 'h', 'r', 's']), random.choice([0,1,2,3]))

        if round_num % 2 == 0:
            round = Round(
                rounds.loc[round_num]["FirstPlayer"], rounds.loc[round_num]["Troef"][0], rounds.loc[round_num]["Gaat"]
            )
            round.set_cards(rounds.loc[round_num]["Cards"])
        else:
            round = Round(
                (rounds.loc[round_num - 1]["FirstPlayer"] + 1) % 4,
                rounds.loc[round_num - 1]["Troef"][0],
                (rounds.loc[round_num - 1]["Gaat"] + 1) % 4,
            )
            cards = rounds.loc[round_num - 1]["Cards"]
            round.set_cards(cards[3:] + cards[:3])

        alpha_player_0.new_round_Round(round)
        alpha_player_2.new_round_Round(round)
        for trick in range(8):
            for j in range(4):

                current_player = round.current_player
                if current_player == 1 or current_player == 3:

                    played_card = rule_player.get_card_good_player(round, current_player)
                    # moves = round.legal_moves()
                    # played_card = random.choice(moves)
                else:
                    tijd = time.time()
                    if current_player == 0:
                        player_hands = round.player_hands
                        legal_moves = round.legal_moves()
                        played_card = alpha_player_0.get_move()
                        played_card = card_untransform(played_card.id, ["k", "h", "r", "s"].index(round.trump_suit))
                    else:
                        player_hands = round.player_hands
                        legal_moves = round.legal_moves()
                        played_card = alpha_player_2.get_move()
                        played_card = card_untransform(played_card.id, ["k", "h", "r", "s"].index(round.trump_suit))
                    alpha_eval_time += time.time() - tijd
                    moves = round.legal_moves()

                    found = False
                    for move in moves:
                        if move.id == played_card:
                            played_card = move
                            found = True
                            break
                    if not found:
                        raise Exception("move not found")

                    # moves = round.legal_moves()
                    # played_card = random.choice(moves)

                round.play_card(played_card)
                move = Card(card_transform(played_card.id, ["k", "h", "r", "s"].index(round.trump_suit)))
                alpha_player_0.update_state(move)
                alpha_player_2.update_state(move)

        for i in range(5):
            mcts_times[i] += alpha_player_0.tijden[i]

        scores_alpha.append(alpha_player_0.state.get_score(0))
        scores_round.append(round.get_score(0))
        if scores_alpha[-1] != scores_round[-1]:
            print("scores_alpha not always equal to scores_round")
            print(scores_alpha[-1], scores_round[-1])
        point_cumulative[0] += round.points[0] + round.meld[0]
        point_cumulative[1] += round.points[1] + round.meld[1]

    return scores_round, point_cumulative, mcts_times, alpha_eval_time / 2

def run_test_multiprocess(
    n_cores: int, opponent: str, total_rounds: int, mcts_params: dict, model_paths: List[str], multiprocessing: bool
):

    rounds_per_process = total_rounds // n_cores
    if rounds_per_process == 0:
        raise Exception("too few rounds to test")
    if opponent == "rule":
        test_function = test_vs_rule_player
    elif opponent == "alphazero":
        test_function = test_vs_alphazero_player
    elif opponent == "pimc":
        test_function = test_vs_rule_player_pimc
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
