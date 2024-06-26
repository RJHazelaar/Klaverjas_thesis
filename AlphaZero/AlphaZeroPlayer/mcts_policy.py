from __future__ import annotations  # To use the class name in the type hinting

import copy
import random
import numpy as np
import time
from scipy.special import softmax

from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.state import State


class MCTS_Node:
    def __init__(self, own_team: bool = True, parent: MCTS_Node = None, move: Card = None, root: bool = False):
        self.children = set()
        self.children_moves = set()
        self.parent = parent
        self.move = move
        self.score = 0
        self.visits = 0
        self.own_team = own_team
        self.q_min = 0
        self.q_max = 162
        self.root = root

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

    def expand(self, new_node):
        move = new_node
        new_node = MCTS_Node(not self.own_team, self, move)
        self.children.add(new_node)
        self.children_moves.add(move)
        return new_node

    def update_max_score(self, score):
        if score > self.q_max:
            self.q_max = score
        #elif score < self.q_min: score always above zero
        #    self.q_min = score

    def normalized_score(self, score):
        return (2 * (score - self.q_min)) / (self.q_max - self.q_min) - 1

    def select_child_puct(self, c: int, simulation, state, model):
        ucbs = []
        return_nodes = []
        legal_moves = list(self.legal_moves)
        if len(legal_moves) == 1:
            nunogniks = 1 #TODO if only 1 move is possible return correct values for performance
        
        # model returns a distribution over 32 features, the cards
        stat = state.to_nparray()
        value, prob_distr = model(np.array([stat])) #32 size array
        prob_distr = prob_distr.numpy().ravel().tolist()

        moves = [a.id for a in legal_moves]
        all_cards = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]
        all_cards_legal = np.in1d(all_cards, moves).astype(int)
        dic = dict(zip(all_cards, prob_distr))
        prob_distr_legal = [0 if x not in dic else dic[x] for x in moves]
        nullen = [0] * len(prob_distr_legal)
        if prob_distr_legal == nullen:
            probabilities_legal = nullen
        else:
            probabilities_legal = prob_distr_legal / np.linalg.norm(prob_distr_legal)

        if self.root:
            probabilities_legal = probabilities_legal #TODO TODO TODO TODO TODO DIRICHLET
        child_prob = dict(zip(moves, probabilities_legal))

        children_moves = []
        children_nodes = []
        for child in self.children:
            children_moves.append(child.move.id)
            children_nodes.append(child)
        children_dict = dict(zip(children_moves, children_nodes))

        for move in moves:
            if move not in children_moves: #Node not added to tree
                return_nodes.append(self)
                if self.own_team:
                    ucbs.append(c * (child_prob[move]))
                else: #TODO gaat nog steeds fout als de trick hierna is afgelopen
                    ucbs.append(c * (child_prob[move]))
            else:
                child = children_dict[move]
                return_nodes.append(child)
                if self.own_team:
                    ucbs.append(self.normalized_score(child.score / child.visits) + c * (child_prob[move]) * (np.sqrt(self.visits) / (1 + child.visits)))
                else:
                    ucbs.append(-self.normalized_score(child.score / child.visits) + c * (child_prob[move]) * (np.sqrt(self.visits) / (1 + child.visits)))
        index_max = np.argmax(np.array([ucbs]))
        return legal_moves[index_max], return_nodes[index_max] #new_node_move, new_node_node



class MCTS:
    def __init__(self, params: dict, model, player_position: int,  **kwargs):
        self.mcts_steps = params["mcts_steps"]
        self.n_of_sims = params["n_of_sims"]
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

    def __call__(self, state: State, training: bool, extra_noise_ratio):
        if self.time_limit != None:
            move, policy = self.mcts_timer(state, training, extra_noise_ratio)
        else:
            move, policy = self.mcts_n_simulations(state, training, extra_noise_ratio)
        return move, policy
    
    def mcts_timer(self, state: State, training: bool, extra_noise_ratio):
        legal_moves = state.legal_moves()
        if len(legal_moves) == 1:
            return next(iter(legal_moves))

        current_state = copy.deepcopy(state)
        current_node = MCTS_Node(root=True)
        # time limit is in ms, time_ns in ns
        ending_time = time.time_ns() + self.time_limit * 1000000
        simulation = -1

        while time.time_ns() < ending_time:
            simulation += 1
            now = time.time()
            # Determination
            current_state.set_determinization()
            self.tijden[0] += time.time() - now
            now = time.time()
            # Selection
            new_node = None
            current_node.set_legal_moves(current_state)
            while (
                not current_state.round_complete() and current_node.legal_moves - current_node.children_moves == set()
            ):
                new_node = current_node.select_child_puct(self.ucb_c, simulation, self.model)
                if new_node not in current_node.children:
                    #Go to expand
                    current_node = current_node
                else:
                    current_node = new_node
                    current_state.do_move(current_node.move, "mcts_move")
                    current_node.set_legal_moves(current_state)
            self.tijden[1] += time.time() - now
            now = time.time()
            # Expansion
            if not current_state.round_complete():
                new_node = current_node.expand(new_node)
                current_node = new_node
                current_state.do_move(current_node.move, "mcts_move")

            self.tijden[2] += time.time() - now
            now = time.time()
            # Simulation
            if not current_state.round_complete():
                sim_score = 0

                if self.model is not None:
                    now2 = time.time()
                    stat = current_state.to_nparray()
                    self.tijden2[0] += time.time() - now2
                    now2 = time.time()
                    arr = np.array([stat])
                    self.tijden2[1] += time.time() - now2
                    now2 = time.time()
                    nn_score = int(self.model(arr)["value_head"])
                    self.tijden2[2] += time.time() - now2
                else:
                    raise Exception("No Model available")
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
        moves = []
        for child in current_node.children:
            visits.append(child.visits)
            children.append(child)
            moves.append(child.move)

        child = children[np.argmax(visits)]

        if training == True:
            visits = np.array(visits) + int(self.mcts_steps * extra_noise_ratio)
            probabilities = visits / np.sum(visits)
            move = np.random.choice(children, p=probabilities).move
        else:
            move = child.move

        # need list of size 32 for (target) policy
        all_cards = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]
        total_visit_count = sum(visits)
        normalized_visits = [x / total_visit_count for x in visits]
        dic = dict(zip(moves, normalized_visits))
        policy = [0 if x not in dic else dic[x] for x in all_cards]

        return move, policy
    
    def mcts_n_simulations(self, state: State, training: bool, extra_noise_ratio):
        legal_moves = state.legal_moves()
        if len(legal_moves) == 1:
            move = next(iter(legal_moves))
            policy = [0] * 32
            policy = [1 if x == move.id else 0 for x in policy]
            return move, policy

        current_state = copy.deepcopy(state)
        current_node = MCTS_Node(root=True)

        for simulation in range(self.mcts_steps):

            now = time.time()
            # Determination
            current_state.set_determinization()
            self.tijden[0] += time.time() - now
            now = time.time()
            # Selection
            new_node = None
            current_node.set_legal_moves(current_state)
            leaf_selected = False
            while (
                not current_state.round_complete() and leaf_selected == False
            ):
                new_node_move, new_node_node = current_node.select_child_puct(self.ucb_c, simulation, current_state, self.model)
                if new_node_move not in current_node.children_moves:
                    #Go to expand
                    current_node = current_node
                    leaf_selected = True
                else:
                    current_node = new_node_node
                    current_state.do_move(current_node.move, "mcts_move")
                    current_node.set_legal_moves(current_state)
            self.tijden[1] += time.time() - now
            now = time.time()
            # Expansion
            if not current_state.round_complete():
                new_node = current_node.expand(new_node_move)
                current_node = new_node
                current_state.do_move(current_node.move, "mcts_move")

            self.tijden[2] += time.time() - now
            now = time.time()
            # Simulation
            if not current_state.round_complete():
                sim_score = 0

                if self.model is not None:
                    now2 = time.time()
                    stat = current_state.to_nparray()
                    self.tijden2[0] += time.time() - now2
                    now2 = time.time()
                    arr = np.array([stat])
                    self.tijden2[1] += time.time() - now2
                    now2 = time.time()
                    nn_score, prob_dist = self.model(arr)
                    nn_score = int(nn_score)
                    self.tijden2[2] += time.time() - now2
                else:
                    raise Exception("No Model available")
            else:
                sim_score = current_state.get_score(self.player_position)
                nn_score = sim_score

            self.tijden[3] += time.time() - now
            now = time.time()
            # Backpropagation
            while current_node.parent is not None:
                current_node.visits += 1
                score = nn_score
                current_node.update_max_score(score)
                current_node.score += score
                current_state.undo_move(current_node.move, True)
                current_node = current_node.parent

            current_node.visits += 1
            current_node.score += (1 - self.nn_scaler) * sim_score + self.nn_scaler * nn_score
            self.tijden[4] += time.time() - now
            now = time.time()

        visits = []
        children = []
        moves = []
        for child in current_node.children:
            visits.append(child.visits)
            children.append(child)
            moves.append(child.move.id)

        child = children[np.argmax(visits)]

        if training == True:
            visits = np.array(visits) + int(self.mcts_steps * extra_noise_ratio)
            probabilities = visits / np.sum(visits)
            move = np.random.choice(children, p=probabilities).move
        else:
            move = child.move

        # need list of size 32 for (target) policy
        all_cards = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]
        total_visit_count = sum(visits)
        normalized_visits = [x / total_visit_count for x in visits]
        dic = dict(zip(moves, normalized_visits))
        policy = [0 if x not in dic else dic[x] for x in all_cards]

        return move, policy