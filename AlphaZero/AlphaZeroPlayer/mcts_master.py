from __future__ import annotations  # To use the class name in the type hinting

import copy
import random
import numpy as np
import time
from collections import Counter

from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.state import State


class MCTS_Node:
    def __init__(self, team, parent: MCTS_Node = None, move: Card = None, root: bool = False):
        self.children = set()
        self.children_moves = set()
        self.parent = parent
        self.move = move
        self.score = 0
        self.visits = 0
        self.team = team
        self.q_min = -162
        self.q_max = 162
        self.root = root
        self.can_follow_suit = [[1] * 4 for i in range(4)]
        # highest by rank: [8, 9, 14, 12, 15, 10, 11, 13][self.value]
        self.highest_trumps = [15,15,15,15] # Trump Jacks

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
        new_node = MCTS_Node(not self.team, self, move)
        self.children.add(new_node)
        self.children_moves.add(move)
        return new_node
    
    def update_max_score(self, score):
        if score > self.q_max:
            self.q_max = score
        elif score < self.q_min:
            self.q_min = score
    
    def normalized_score(self, score):
        return (2 * (score - self.q_min)) / (self.q_max - self.q_min) - 1

    def select_child_puct(self, c, simulation, root_team, state, model):
        #TODO unhardcode this
        c_2 = 19652
        ucbs = []
        return_nodes = []
        legal_moves = list(self.legal_moves)

        # Return the only legal move from the state #TODO what node to return
        if len(legal_moves) == 1:
            children_moves = []
            children_nodes = []
            return_node = self
            for child in self.children:
                if child.move.id == legal_moves[0].id:
                    return_node = child
            return legal_moves[0], return_node 
        
        # model returns a distribution over 32 features, the cards
        stat = state.to_nparray_alt()
        if(state.current_player == state.own_position):
            stat = state.to_nparray_alt()
        else:
            stat = state.to_nparray_alt_op()


        value, prob_distr = model(np.array([stat])) #32 size array
        prob_distr = prob_distr.numpy().ravel().tolist()

        moves = [a.id for a in legal_moves]
        all_cards = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]

        dic = dict(zip(all_cards, prob_distr))
        # Remove probabilities of illegal moves
        prob_distr_legal = [0 if x not in dic else dic[x] for x in moves]
        zeroes = [0] * len(prob_distr_legal)
        # Renormalize probabilities
        if prob_distr_legal == zeroes:
            probabilities_legal = zeroes
        else:
            probabilities_legal = prob_distr_legal / np.linalg.norm(prob_distr_legal)

        # Add Dirichlet noise for added exploration from root during training
        if self.root:
            dirichlet_alpha = 0.03
            dirichlet_epsilon = 0.25
            dirichlet_noise = [dirichlet_alpha] * len(probabilities_legal)
            for index, prob in enumerate(probabilities_legal):
                probabilities_legal[index] = (1-dirichlet_epsilon)*prob + dirichlet_epsilon * dirichlet_noise[index]

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
                ucbs.append(c * (child_prob[move]))
            else:
                child = children_dict[move]
                return_nodes.append(child)
                if self.team == root_team:
                    ucbs.append(self.normalized_score(child.score / child.visits) + (child_prob[move]) * (np.sqrt(self.visits) / (1 + child.visits)) * (c + np.log((self.visits + c_2 + 1)/(c_2))))
                else:
                    ucbs.append(-self.normalized_score(child.score / child.visits) + (child_prob[move]) * (np.sqrt(self.visits) / (1 + child.visits)) * (c + np.log((self.visits + c_2 + 1)/(c_2))))
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
        try: 
            self.steps_per_determinization = params["steps_per_determinization"]
        except:
            print("no steps per determinization")
        try: 
            self.dirichlet_noise = params["dirichlet_noise"]
        except:
            self.dirichlet_noise = 0

    def __call__(self, state: State, training: bool, extra_noise_ratio):
        if self.time_limit != None:
            move, policy = self.mcts_timer(state, training, extra_noise_ratio)
        else:
            move, policy = self.pimc_call(state, training, extra_noise_ratio)
        return move, policy
    
    def pimc_call(self, state, training, extra_noise_ratio):
        all_cards = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]

        legal_moves = state.legal_moves()
        if len(legal_moves) == 1:
            move = next(iter(legal_moves))
            policy = [1 if x == move.id else 0 for x in all_cards]
            return move, policy
        
        # for fixed order of moves
        legal_moves_list = list(legal_moves)
        legal_moves_list_id = [leg_m.id for leg_m in legal_moves_list]

        combined_policy = np.zeros(len(legal_moves))
        for determinization in range(self.mcts_steps // self.steps_per_determinization):
            move, policy_dict = self.mcts_n_simulations(state, training, extra_noise_ratio, self.steps_per_determinization)
            # Moves might not always be in the same order
            policy = [policy_dict[x] for x in legal_moves_list]            
            combined_policy += np.array(policy)
        
        if self.mcts_steps % self.steps_per_determinization > len(legal_moves):
            move, policy_dict = self.mcts_n_simulations(state, training, extra_noise_ratio, self.mcts_steps % self.steps_per_determinization)
            # Moves might not always be in the same order
            policy = [policy_dict[x] for x in legal_moves_list]            
            combined_policy += np.array(policy)
        
        visits = combined_policy.tolist()
        child = legal_moves_list[np.argmax(visits)]

        probabilities = visits / np.sum(visits)
        if training == True:
            visits = np.array(visits) + int(self.mcts_steps * extra_noise_ratio)
            move = np.random.choice(legal_moves_list, p=probabilities)
        else:
            move = child

        # policy is over all 32 possible moves, need list of size 32 for (target) policy
        dic = dict(zip(legal_moves_list_id, probabilities))
        policy = [0 if x not in dic else dic[x] for x in all_cards]

        return move, policy

    def mcts_timer(self, state: State, training: bool, extra_noise_ratio):
        legal_moves = state.legal_moves()
        if len(legal_moves) == 1:
            return next(iter(legal_moves))

        current_state = copy.deepcopy(state)
        current_node = MCTS_Node()
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
                for _ in range(self.n_of_sims):
                    children = []

                    # Do random moves until round is complete
                    while not current_state.round_complete():

                        move = random.choice(list(current_state.legal_moves()))
                        children.append(move)
                        current_state.do_move(move, "simulation")

                    # Add score to points
                    sim_score += current_state.get_score(self.player_position)

                    # Undo moves
                    children.reverse()
                    for move in children:
                        current_state.undo_move(move, False)

                # Average the score
                if self.n_of_sims > 0:
                    sim_score /= self.n_of_sims

                if self.model is not None:
                    now2 = time.time()
                    stat = current_state.to_nparray_alt()
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
    
    def mcts_n_simulations(self, state: State, training: bool, extra_noise_ratio, steps):
        legal_moves = state.legal_moves()
        if len(legal_moves) == 1: #TODO Does not return policy, but having only 1 legal move is resolved earlier in pimc_call
            return next(iter(legal_moves)).id

        current_state = copy.deepcopy(state)
        # Team of root player
        root_team = current_state.current_player % 2
        root_player = current_state.current_player
        root_info_suits = copy.deepcopy(state.can_follow_suit)
        root_highest_trumps = copy.deepcopy(state.highest_trumps)
        current_node = MCTS_Node(team = root_team)

        for simulation in range(steps):
            current_state.reset_information_set(root_info_suits, root_highest_trumps)
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
                new_node_move, new_node_node = current_node.select_child_puct(self.ucb_c, simulation, root_team, current_state, self.model)
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
                current_state.do_move(new_node_move, "mcts_move")
                new_node = current_node.expand(new_node_move)
                current_node = new_node
                current_node.team = current_state.current_player % 2
                current_node.can_follow_suit = copy.deepcopy(current_state.can_follow_suit)
                current_node.highest_trumps = copy.deepcopy(current_state.highest_trumps)

            self.tijden[2] += time.time() - now
            now = time.time()
            # Simulation
            if not current_state.round_complete():
                sim_score = 0

                if self.model is not None:
                    now2 = time.time()
                    if(current_state.current_player == root_player):
                        stat = current_state.to_nparray_alt()
                    else:
                        stat = current_state.to_nparray_alt_op()
                    self.tijden2[0] += time.time() - now2
                    now2 = time.time()
                    arr = np.array([stat])
                    self.tijden2[1] += time.time() - now2
                    now2 = time.time()
                    nn_score, prob_dist = self.model(arr)
                    if (current_state.current_player % 2 != root_team):
                        nn_score = int(-nn_score)
                    else:
                        nn_score = int(nn_score)
                    self.tijden2[2] += time.time() - now2

                if self.nn_scaler < 1: 
                    for _ in range(self.n_of_sims):
                        children = []

                        # Do random moves until round is complete
                        while not current_state.round_complete():

                            move = random.choice(list(current_state.legal_moves()))
                            children.append(move)
                            current_state.do_move(move, "simulation")

                        # Add score to points
                        sim_score += current_state.get_score(self.player_position) # Score from perspective of root node player

                        # Undo moves
                        children.reverse()
                        for move in children:
                            current_state.undo_move(move, False)                
                
                # Average the score
                if self.n_of_sims > 0:
                    sim_score /= self.n_of_sims



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
            # Reset Information Set to root set

            current_node.visits += 1
            current_node.score += nn_score
            self.tijden[4] += time.time() - now
            now = time.time()

        visits = []
        children = []
        moves = []

        for child in current_node.children:
            visits.append(child.visits)
            children.append(child)
            moves.append(child.move)

        best_move = children[np.argmax(visits)].move

        # If not all legal moves have been visited from root
        legal_moves_list = list(legal_moves)
        if len(moves) < len(legal_moves_list):
            for legal_m in legal_moves_list:
                if legal_m not in moves:
                    moves.append(legal_m)
                    visits.append(0)

        if training == True and 1 == 0:
            visits = np.array(visits) + int(self.mcts_steps * extra_noise_ratio)
            probabilities = visits / np.sum(visits)
            best_move = np.random.choice(moves, p=probabilities)


        policy_dict = dict(zip(moves, visits))
        return best_move.id, policy_dict
