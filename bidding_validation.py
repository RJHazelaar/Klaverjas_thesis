from __future__ import annotations  # To use the class name in the type hinting

import os
import time
import tensorflow as tf
import pandas as pd
import sys
import numpy as np
import random

from multiprocessing import Pool, get_context
from typing import List

from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.helper import card_transform, card_untransform
from AlphaZero.AlphaZeroPlayer.alphazero_player import AlphaZero_player
from AlphaZero.AlphaZeroPlayer.alphazero_player_heavyrollout import AlphaZero_player_heavyrollout
from Lennard.rounds import Round
from Lennard.rule_based_agent import Rule_player

import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../")))
sys.path.append(parent_dir)
data_dir = parent_dir+"/Klaverjas_thesis-main/Data/SL_Data/originalDB.csv"
model_path = parent_dir+"/Klaverjas_thesis/Experiments/voor_nu/bidding_network_policy_99.h5"

def validation_test(num_rounds: int, model_paths):
    # random.seed(13)
    alpha_eval_time = 0
    mcts_times = [0, 0, 0, 0, 0]
    total_rounds = 0
    right_predictions = 0

    rounds = pd.read_csv(data_dir, delimiter=";", low_memory=False, converters={"Cards": pd.eval})

    rule_player = Rule_player()
    mcts_params = {
        "mcts_steps": 200,
        "n_of_sims": 1,
        "nn_scaler": 0,
        "ucb_c": 200,
    }

    print(model_paths[0])
    model = None
    if model_paths[0] is not None:
        try:
            model = tf.keras.models.load_model(model_path)
        except:
            print("model not found")
            raise Exception("model not found")

    alpha_player_0 = AlphaZero_player(0, mcts_params, model)
    alpha_player_2 = AlphaZero_player(2, mcts_params, model)

    for round_num in range(0, num_rounds):
        # round = Round((starting_player + 1) % 4, random.choice(['k', 'h', 'r', 's']), random.choice([0,1,2,3]))
        total_rounds += 1

        round = Round(rounds.loc[round_num]["FirstPlayer"], rounds.loc[round_num]["Troef"][0], rounds.loc[round_num]["Gaat"])
        round.set_cards(rounds.loc[round_num]["Cards"])
        
        options = ["k","h","r","s"]
        trump_from_data = round.trump_suit
        declarer_from_data = round.declarer

        predicted_declarer = None
        predicted_trump = "p"


        declarer = round.starting_player
        starting_player = round.starting_player

        #print(f"Ground truth from data, trump: {trump_from_data}, declarer{declarer_from_data}")
        #print(f"Predictions from network, trump: {predicted_trump}, declarer{predicted_declarer}")

        passes = 0
        bidding_order = list(range(declarer, 4)) + list(range(0, declarer))
        for bidder in bidding_order:
            trump_scores = []
            possible_trump_suit = "k"
            for index, trump in enumerate(["k","h","r","s"]): #starting_player==declarer in first bid
                input_vector = round.hand_to_input_vector_alt(trump, bidder, starting_player)
                output = model(input_vector)

                trump_scores.append(output)
            
            possible_trump_suit = options[np.argmax(trump_scores)]
            # If player predicts a win by a slight margin
            if trump_scores[np.argmax(trump_scores)] > 0:
                predicted_declarer = bidder
                predicted_trump = possible_trump_suit
                break
            else:
                passes += 1

        
        if passes == 4: # First declarer forced to make a decision, can't pass
            trump_scores = []
            for index, trump in enumerate(["k","h","r","s"]): #starting_player==declarer in first bid
                input_vector = round.hand_to_input_vector_alt(trump, bidder, starting_player)
                output = model(input_vector)
                trump_scores.append(output)
            
            predicted_trump = options[np.argmax(trump_scores)] #TODO Just use the previous output

        if trump_from_data == predicted_trump and declarer_from_data == predicted_declarer:
            right_predictions += 1 

    return right_predictions / total_rounds

def run_test_multiprocess(n_cores: int, model_path):

    rounds_per_process = 87218 #length of dataset
    scores_round = []
    points_cumulative = [0, 0]
    mcts_times = [0, 0, 0, 0, 0]
    alpha_eval_time = 0

    total_score = validation_test(rounds_per_process, model_path)

    alpha_eval_time = round(alpha_eval_time * 1000, 1)  # convert to ms and round to 1 decimal
    temp = None
    return total_score

def main():
    total_score = run_test_multiprocess(1, model_path)
    print(total_score)

if __name__ == "__main__":
    main()