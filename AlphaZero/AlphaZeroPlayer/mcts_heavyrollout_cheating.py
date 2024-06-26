from __future__ import annotations
import array  # To use the class name in the type hinting

import copy
import random
import numpy as np
import time

from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.state import State


class MCTS_Node:
    def __init__(self, own_team: bool = True, parent: MCTS_Node = None, move: Card = None):
        self.children = set()
        self.children_moves = set()
        self.parent = parent
        self.move = move
        self.score = 0
        self.visits = 0
        self.own_team = own_team

    def __repr__(self) -> str:
        return f"Node({self.move}, {self.parent.move}, {self.score}, {self.visits})"

    def __eq__(self, other: MCTS_Node) -> bool:
        # raise NotImplementedError
        return self.move == other.move

    def __hash__(self) -> int:
        # raise NotImplementedError
        return hash(self.move)

    def set_legal_moves(self, state: State):
        self.legal_moves = state.legal_moves()

    def expand(self):
        move = random.choice(list(self.legal_moves - self.children_moves))
        new_node = MCTS_Node(not self.own_team, self, move)
        self.children.add(new_node)
        self.children_moves.add(move)
        return new_node

    def select_child_ucb(self, c: int, simulation) -> MCTS_Node:
        ucbs = []
        legal_children = [child for child in self.children if child.move in self.legal_moves]
        for child in legal_children:
            if child.visits == 0:
                return child
            if self.own_team:
                ucbs.append(child.score / child.visits + c * np.sqrt(np.log(simulation) / child.visits))
            else:
                ucbs.append(-child.score / child.visits + c * np.sqrt(np.log(simulation) / child.visits))
        index_max = np.argmax(np.array([ucbs]))
        return legal_children[index_max]


class MCTS:
    def __init__(self, params: dict, model, player_position: int, **kwargs):
        self.mcts_steps = params["mcts_steps"]
        self.n_of_sims = params["n_of_sims"]
        # TODO Op 1 zetten om performance ok te houden, meerdere sims hebben geen zin
        self.n_of_sims = 1
        self.ucb_c = params["ucb_c"]
        self.nn_scaler = params["nn_scaler"]
        self.player_position = player_position
        self.model = model
        self.tijden = [0, 0, 0, 0, 0]
        self.tijden2 = [0, 0, 0]
        try:
            self.time_limit = params["time_limit"]
        except:
            self.time_limit = None
        try:
            self.e_cheat = params["e_cheat"]
        except:
            self.e_cheat = 0 # 0% chance of using the cheating determinization

    def __call__(self, state: State, training: bool, extra_noise_ratio, player_hands):
        if self.time_limit != None:
            move = self.mcts_timer(state, training, extra_noise_ratio, player_hands)
        else:
            move = self.mcts_n_simulations(state, training, extra_noise_ratio, player_hands)
        return move

    def mcts_timer(self, state: State, training: bool, extra_noise_ratio, player_hands):
        legal_moves = state.legal_moves()
        if len(legal_moves) == 1:
            return next(iter(legal_moves))

        current_state = copy.deepcopy(state)
        current_node = MCTS_Node()
        # time limit is in ms, time_ns in ns
        ending_time = time.time_ns() + self.time_limit * 1000000
        simulation = -1
        #TODO Testing Purposes, Remove
        #print("staring time:", time.time_ns())
        #print("ending time:", ending_time)
        #t1_start = time.perf_counter()
        while time.time_ns() < ending_time:
            simulation += 1
            now = time.time()
            # Determination
            if random.random() < self.e_cheat:
                current_state.set_determinization_cheat(player_hands)
            else:    
                current_state.set_determinization()
            self.tijden[0] += time.time() - now
            now = time.time()
            # Selection
            current_node.set_legal_moves(current_state)
            while (
                not current_state.round_complete() and current_node.legal_moves - current_node.children_moves == set()
            ):
                current_node = current_node.select_child_ucb(self.ucb_c, simulation)
                current_state.do_move(current_node.move, "mcts_move")
                current_node.set_legal_moves(current_state)
            self.tijden[1] += time.time() - now
            now = time.time()
            # Expansion
            if not current_state.round_complete():
                new_node = current_node.expand()
                current_node = new_node
                current_state.do_move(current_node.move, "mcts_move")

            self.tijden[2] += time.time() - now
            now = time.time()
            # Simulation
            if not current_state.round_complete():
                sim_score = 0
                #############################################################################################################################################
                for _ in range(self.n_of_sims):
                    children = []

                    while not current_state.round_complete():
                        #TODO bekijk legal_moves method om wat mooiere implementatie te maken          
                        newmove = self.get_card_good_player(current_state)
                        # Convert from <class 'Lennard.deck.Card' to <class 'AlphaZero.AlphaZeroPlayer.Klaverjas.card.Card'>
                        move = Card(newmove.id)
                        children.append(move)
                        current_state.do_move(move, "simulation")                       

                    # Add score to points
                    sim_score += current_state.get_score(self.player_position)

                    # Undo moves
                    children.reverse()
                    for move in children:
                        current_state.undo_move(move, False)

                #############################################################################################################################################
                # Average the score
                if self.n_of_sims > 0:
                    sim_score /= self.n_of_sims

                if self.model is not None:
                    now2 = time.time()
                    stat = current_state.to_nparray()
                    self.tijden2[0] += time.time() - now2
                    now2 = time.time()
                    arr = np.array([stat])
                    self.tijden2[1] += time.time() - now2
                    now2 = time.time()
                    nn_score = int(self.model(arr))
                    self.tijden2[2] += time.time() - now2
                else:
                    nn_score = 0
            else:
                sim_score = current_state.get_score(self.player_position)
                nn_score = sim_score

            self.tijden[3] += time.time() - now
            now = time.time()
            # Backpropagation
            while current_node.parent is not None:
                current_node.visits += 1
                current_node.score += (1 - self.nn_scaler) * sim_score + self.nn_scaler * nn_score
                current_state.undo_move(current_node.move, True)
                current_node = current_node.parent

            current_node.visits += 1
            current_node.score += (1 - self.nn_scaler) * sim_score + self.nn_scaler * nn_score
            self.tijden[4] += time.time() - now
            now = time.time()

        #TODO Remove, testing purposes
        #print("we did done ended")
        #t1_stop = time.perf_counter()
        #print("elapsed time:", t1_stop, t1_start)
        #print("elapsed time during loop:", t1_stop-t1_start)

        visits = []
        children = []
        for child in current_node.children:
            visits.append(child.visits)
            children.append(child)

        child = children[np.argmax(visits)]

        if training == True:
            visits = np.array(visits) + int(self.mcts_steps * extra_noise_ratio)
            probabilities = visits / np.sum(visits)
            move = np.random.choice(children, p=probabilities).move
        else:
            move = child.move

        return move


    def mcts_n_simulations(self, state: State, training: bool, extra_noise_ratio, player_hands):
        legal_moves = state.legal_moves()
        if len(legal_moves) == 1:
            return next(iter(legal_moves))

        current_state = copy.deepcopy(state)
        current_node = MCTS_Node()
        for simulation in range(self.mcts_steps):

            now = time.time()
            # Determination
            if random.random() < self.e_cheat:
                current_state.set_determinization_cheat(player_hands)
            else:    
                current_state.set_determinization()
            self.tijden[0] += time.time() - now
            now = time.time()
            # Selection
            current_node.set_legal_moves(current_state)
            while (
                not current_state.round_complete() and current_node.legal_moves - current_node.children_moves == set()
            ):
                current_node = current_node.select_child_ucb(self.ucb_c, simulation)
                current_state.do_move(current_node.move, "mcts_move")
                current_node.set_legal_moves(current_state)
            self.tijden[1] += time.time() - now
            now = time.time()
            # Expansion
            if not current_state.round_complete():
                new_node = current_node.expand()
                current_node = new_node
                current_state.do_move(current_node.move, "mcts_move")

            self.tijden[2] += time.time() - now
            now = time.time()
            # Simulation
            if not current_state.round_complete():
                sim_score = 0
                #############################################################################################################################################
                for _ in range(self.n_of_sims):
                    children = []

                    while not current_state.round_complete():
                        #TODO bekijk legal_moves method om wat mooiere implementatie te maken          
                        newmove = self.get_card_good_player(current_state)
                        # Convert from <class 'Lennard.deck.Card' to <class 'AlphaZero.AlphaZeroPlayer.Klaverjas.card.Card'>
                        move = Card(newmove.id)
                        children.append(move)
                        current_state.do_move(move, "simulation")                       

                    # Add score to points
                    sim_score += current_state.get_score(self.player_position)

                    # Undo moves
                    children.reverse()
                    for move in children:
                        current_state.undo_move(move, False)

                #############################################################################################################################################
                # Average the score
                if self.n_of_sims > 0:
                    sim_score /= self.n_of_sims

                if self.model is not None:
                    now2 = time.time()
                    stat = current_state.to_nparray()
                    self.tijden2[0] += time.time() - now2
                    now2 = time.time()
                    arr = np.array([stat])
                    self.tijden2[1] += time.time() - now2
                    now2 = time.time()
                    nn_score = int(self.model(arr))
                    self.tijden2[2] += time.time() - now2
                else:
                    nn_score = 0
            else:
                sim_score = current_state.get_score(self.player_position)
                nn_score = sim_score

            self.tijden[3] += time.time() - now
            now = time.time()
            # Backpropagation
            while current_node.parent is not None:
                current_node.visits += 1
                current_node.score += (1 - self.nn_scaler) * sim_score + self.nn_scaler * nn_score
                current_state.undo_move(current_node.move, True)
                current_node = current_node.parent

            current_node.visits += 1
            current_node.score += (1 - self.nn_scaler) * sim_score + self.nn_scaler * nn_score
            self.tijden[4] += time.time() - now
            now = time.time()

        visits = []
        children = []
        for child in current_node.children:
            visits.append(child.visits)
            children.append(child)

        child = children[np.argmax(visits)]

        if training == True:
            visits = np.array(visits) + int(self.mcts_steps * extra_noise_ratio)
            probabilities = visits / np.sum(visits)
            move = np.random.choice(children, p=probabilities).move
        else:
            move = child.move

        return move
    




    #Returns the card to play according to the rule based agent
    def get_card_good_player(self, cur_state):
        player = cur_state.current_player
        trick = cur_state.tricks[-1]
        round = cur_state.round
        trump = round.trump_suit
        legal_moves = list(cur_state.legal_moves())

        #When only 1 move is possible, select this move
        if len(legal_moves) == 1:
            return legal_moves[0]
    
        # Get highest value of highest value card still in the game for each suit
        all_cards = list(cur_state.possible_cards[0] | cur_state.possible_cards[1] | cur_state.possible_cards[2] | cur_state.possible_cards[3])

        # TODO Change this hardcoded method into more elegant function

        # Split cards intro respective suits
        trump_cards = [x for x in all_cards if x.id < 8]
        suit1_cards = [x for x in all_cards if x.id > 9 and x.id < 18]
        suit2_cards = [x for x in all_cards if x.id > 19 and x.id < 28]
        suit3_cards = [x for x in all_cards if x.id > 29 and x.id < 38]
        
        trump_cards = sorted(trump_cards, key=lambda card: card.order())
        suit1_cards = sorted(suit1_cards, key=lambda card: card.order())
        suit2_cards = sorted(suit2_cards, key=lambda card: card.order())
        suit3_cards = sorted(suit3_cards, key=lambda card: card.order())

        ordered_cards = [trump_cards, suit1_cards, suit2_cards, suit3_cards]
        dummy_card = Card(-1) # Card without value for exception handling
        for cards in ordered_cards:
            if len(cards) == 0:
                cards.append(dummy_card) # list without cards for handling index out of range
        
        cant_follow = 0
        if len(trick.cards) == 0:
            for card in legal_moves:
                if card.id == ordered_cards[card.suit][-1].id: # Use id instead of value, legal_moves might include same value cards from different suit
                    return card
            return self.get_lowest_card(legal_moves)

        if legal_moves[0].suit == trick.cards[0].suit:
            cant_follow = 1
            
        if len(trick.cards) == 1:
            for card in legal_moves:
                if card.id == ordered_cards[trick.leading_suit()][-1].id:   #TODO CHECK VOOR LIST INDEX OUT OF RANGE
                    if card.order() > trick.cards[0].order(): # Is card better than leading card
                        return card
            return self.get_lowest_card(legal_moves)

        if len(trick.cards) == 2:
            if cant_follow:
                return self.get_lowest_card(legal_moves)

            # In turn 3 the player in 1st position is always teammate
            if trick.cards[0].id == ordered_cards[trick.leading_suit()][-1].id:
                return self.get_highest_card(legal_moves)

            # ???? TODO Waar komt deze regel vandaan ????
            for card in legal_moves:
                if card.id == ordered_cards[trick.leading_suit()][-1].id:
                    return card
            # ???? TODO ????

            return self.get_lowest_card(legal_moves)

        else:
            
            if trick.winner() %2 == trick.to_play() %2:
                return self.get_highest_card(legal_moves)

            highest_card = trick.cards[0]
            for card in trick.cards:
                if card.order() > highest_card.order():
                    highest_card = card
                    
            for card in legal_moves:
                if card.order() > highest_card.order():
                    return card

            return self.get_lowest_card(legal_moves)
    
    def get_lowest_card(self, cards):
        lowest_points = 21
        for card in cards:
            if card.points() < lowest_points:
                lowest_card = card
                lowest_points = card.points()
        return lowest_card

    def get_highest_card(self, cards):
        highest_points = -1
        for card in cards:
            if card.points() > highest_points:
                highest_card = card
                highest_points = card.points()
        return highest_card
    
    # TODO Clean up code by instead making highest_card function work with card ranks and suits