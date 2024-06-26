from __future__ import annotations  # To use the class name in the type hinting

from Lennard.rounds import Round
from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.state import State
from AlphaZero.AlphaZeroPlayer.mcts_test import MCTS


class AlphaZero_player_test:
    def __init__(self, player_position: int, mcts_params: dict, model, **kwargs):
        self.player_position = player_position
        self.model = model
        self.mcts = MCTS(mcts_params, model, player_position, time_limit=kwargs.get('time_limit'))
        self.tijden = [0, 0, 0, 0, 0]
        self.state = None

    def new_round_Round(self, round: Round):
        if self.state is not None:
            for i in range(len(self.tijden)):
                self.tijden[i] += self.state.tijden[i]
        self.state = State(self.player_position)
        self.state.init_from_Round(round)

    def new_round_klaverlive(self, hand, starting_player, declaring_team):
        self.state = State(self.player_position)
        self.state.init_from_klaverlive(hand, starting_player, declaring_team)

    def update_state(self, move: Card):
        self.state.do_move(move)

    def get_move(self, training: bool = False, extra_noise_ratio=0):
        if self.player_position != self.state.current_player:
            print(self.state.current_player, self.player_position)
            raise Exception("Not this player's turn")
        return self.mcts(self.state, training, extra_noise_ratio)
