from __future__ import annotations  # To use the class name in the type hinting

import random
import time
import numpy as np
import copy

from Lennard.rounds import Round
from AlphaZero.AlphaZeroPlayer.Klaverjas.trick import Trick
from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.helper import card_transform, team


class State:
    def __init__(self, own_position: int) -> None:
        self.own_position = own_position
        self.pit_check = 0
        self.cards_left = [8, 8, 8, 8]
        self.final_score = [0, 0]
        self.points = [0, 0]
        self.meld = [0, 0]
        self.tijden = [0, 0, 0, 0, 0]
        # What player can follow what suit according to public information
        # [player][suit]
        self.can_follow_suit = [[1] * 4 for i in range(4)]
        # highest by rank: [8, 9, 14, 12, 15, 10, 11, 13][self.value]
        self.highest_trumps = [15,15,15,15] # Trump Jacks

    def init_from_Round(self, round: Round):
        self.round = round
        self.current_player = round.current_player
        self.declaring_team = round.declaring_team
        self._trump_suit = round.trump_suit

        # The hand of the transformed to the game state representation
        own_hand_as_id = [
            card_transform(card.id, ["k", "h", "r", "s"].index(round.trump_suit))
            for card in round.player_hands[self.own_position]
        ]

        not_own_hand_as_id = set([suit * 10 + value for suit in range(4) for value in range(8)]) - set(own_hand_as_id)

        self.hands = [set() for _ in range(4)]
        self.hands[self.own_position] = set([Card(id) for id in own_hand_as_id])

        self.possible_cards = [set([Card(id) for id in not_own_hand_as_id]) for _ in range(4)]
        self.possible_cards[self.own_position] = set([Card(id) for id in own_hand_as_id])
        self.removed_cards = [set() for _ in range(4)]

        self.tricks = [Trick(self.current_player)]

        # What player can follow what suit according to public information
        self.can_follow_suit = [[1] * 4 for i in range(4)]
        # highest by rank: [8, 9, 14, 12, 15, 10, 11, 13][self.value]
        self.highest_trumps = [15,15,15,15] # Trump Jacks

    def init_from_klaverlive(self, own_hand, starting_player, declaring_team):
        self.current_player = starting_player
        self.declaring_team = declaring_team

        # The hand of the transformed to the game state representation
        own_hand_as_id = own_hand

        not_own_hand_as_id = set([suit * 10 + value for suit in range(4) for value in range(8)]) - set(own_hand_as_id)

        self.hands = [set() for _ in range(4)]
        self.hands[self.own_position] = set([Card(id) for id in own_hand_as_id])

        self.possible_cards = [set([Card(id) for id in not_own_hand_as_id]) for _ in range(4)]
        self.possible_cards[self.own_position] = set([Card(id) for id in own_hand_as_id])
        self.removed_cards = [set() for _ in range(4)]

        self.tricks = [Trick(self.current_player)]

        # What player can follow what suit according to public information
        self.can_follow_suit = [[1] * 4 for i in range(4)]
        # highest by rank: [8, 9, 14, 12, 15, 10, 11, 13][self.value]
        self.highest_trumps = [15,15,15,15] # Trump Jacks

    def set_determinization(self):
        other_players = [0, 1, 2, 3]
        other_players.pop(self.own_position)

        possible_cards = [self.possible_cards[i].copy() for i in other_players]

        cards_left = self.cards_left.copy()
        cards_left.pop(self.own_position)


        all_cards = list(possible_cards[0] | possible_cards[1] | possible_cards[2])
        random.shuffle(all_cards)

        all_cards2 = []
        for card in all_cards:
            players = []
            if card in possible_cards[0]:
                players.append(0)
            if card in possible_cards[1]:
                players.append(1)
            if card in possible_cards[2]:
                players.append(2)
            all_cards2.append((card, players))

        hands = [set(), set(), set()]
        if not self.find_determinization(all_cards2, hands, possible_cards, cards_left):
            raise Exception("No determinization found")

        for index, player in enumerate(other_players):
            self.hands[player] = hands[index]

    def reset_determinization(self, hands):
        all_hands = [set() for _ in range(4)]
        for player, hand in enumerate(hands):
            for card in hand:
                all_hands[player].add(card)
            self.hands[player] = all_hands[player]
        return

    def get_determinization(self):
        return self.hands

    def find_determinization(
        self,
        all_cards: list[tuple],
        hands: list[set[Card]],
        possible_cards: list[set[Card]],
        num_cards_to_add: list[int],
    ):
        if all_cards == []:
            return True
        else:
            stop = False
            all_cards_copy = all_cards.copy()
            card = all_cards_copy.pop(random.choice(range(len(all_cards_copy))))
            random.shuffle(card[1])
            for player in card[1]:
                num_cards_to_add[player] -= 1

                if num_cards_to_add[player] < 0:
                    num_cards_to_add[player] += 1
                    continue

                for other_player in card[1]:
                    possible_cards[other_player].remove(card[0])
                    if len(possible_cards[other_player]) < num_cards_to_add[other_player]:
                        stop = True
                        break

                if stop:
                    stop = False
                    num_cards_to_add[player] += 1
                    for other_player in card[1]:
                        possible_cards[other_player].add(card[0])
                    continue

                hands[player].add(card[0])

                if self.find_determinization(all_cards_copy, hands, possible_cards, num_cards_to_add):
                    return True

                num_cards_to_add[player] += 1

                for other_player in card[1]:
                    possible_cards[other_player].add(card[0])

                hands[player].remove(card[0])

            return False

    def set_determinization_cheat(self, player_hands):
        other_players = [0, 1, 2, 3]
        other_players.pop(self.own_position)

        possible_cards = [self.possible_cards[i].copy() for i in other_players] #WRONG?

        other_player_hands = player_hands.copy()
        other_player_hands.pop(self.own_position)

        cards_left = self.cards_left.copy()
        cards_left.pop(self.own_position)

        all_cards = list(possible_cards[0] | possible_cards[1] | possible_cards[2]) #WRONG?
        all_cards2 = []

        for card in all_cards:
            players = []
            if card.id in other_player_hands[0]:
                players.append(0)
            elif card.id in other_player_hands[1]:
                players.append(1)
            elif card.id in other_player_hands[2]:
                players.append(2)
            all_cards2.append((card, players))


        all_cards2 = []
        for player in range(len(other_players)):
            for id in other_player_hands[player]: 
                all_cards2.append((Card(id), [player]))

        hands = [set(), set(), set()]
        if not self.find_determinization(all_cards2, hands, possible_cards, cards_left):
            raise Exception("No determinization found")
        
        for index, player in enumerate(other_players):
            self.hands[player] = hands[index]

    def update_possible_cards(self, played_card: Card, reversable: bool):

        # remove played card from all players possible_cards
        for player in range(4):
            if reversable and played_card in self.possible_cards[player]:
                self.removed_cards[player] |= {played_card}
            self.possible_cards[player].discard(played_card)

        leading_suit = self.tricks[-1].leading_suit()
        if leading_suit is None:
            return
 
        if self.current_player == self.own_position:
            # Record what information other players can infer
            
            if played_card.suit == 0:
                if (highest_trump_order := self.tricks[-1].highest_trump().order()) > played_card.order():
                    # update public information
                    self.highest_trumps[self.current_player] = highest_trump_order
            if played_card.suit != leading_suit:
                # update public information
                self.can_follow_suit[self.current_player][leading_suit] = 0
                # remove all cards of the leading suit from the current player
                if played_card.suit != 0:
                    # update public information
                    self.can_follow_suit[self.current_player][0] = 0
            return



        if played_card.suit == 0:
            if (highest_trump_order := self.tricks[-1].highest_trump().order()) > played_card.order():
                # update public information
                self.highest_trumps[self.current_player] = highest_trump_order
                # remove all trump cards higher then the highest trump card from the current player
                cards_to_remove = {
                    card
                    for card in self.possible_cards[self.current_player]
                    if card.id in [0, 1, 5, 6, 3, 7, 2, 4][highest_trump_order - 8 :]
                }
                if reversable:
                    if cards_to_remove != cards_to_remove & self.possible_cards[self.current_player]:
                        raise Exception("Klopt nie heh")
                    self.removed_cards[self.current_player] |= cards_to_remove
                self.possible_cards[self.current_player] -= cards_to_remove

        if played_card.suit != leading_suit:
            # update public information
            self.can_follow_suit[self.current_player][leading_suit] = 0
            # remove all cards of the leading suit from the current player
            cards_to_remove = {Card(leading_suit * 10 + i) for i in range(8)}
            if reversable:
                self.removed_cards[self.current_player] |= cards_to_remove & self.possible_cards[self.current_player]
            self.possible_cards[self.current_player] -= cards_to_remove

            if played_card.suit != 0:
                # update public information
                self.can_follow_suit[self.current_player][0] = 0
                # remove all trumps from the current player
                cards_to_remove = {
                    card for card in self.possible_cards[self.current_player] if card.id in {0, 1, 2, 3, 4, 5, 6, 7}
                }
                if reversable:
                    if cards_to_remove != cards_to_remove & self.possible_cards[self.current_player]:
                        raise Exception("Klopt nie heh")
                    self.removed_cards[self.current_player] |= cards_to_remove
                self.possible_cards[self.current_player] -= cards_to_remove

    def reset_information_set(self, info_suits, highest_trumps):

        self.can_follow_suit = copy.deepcopy(info_suits)
        self.highest_trumps = copy.deepcopy(highest_trumps)
        for index_i, i in enumerate(info_suits):
            self.highest_trumps[index_i] = highest_trumps[index_i]
            for index_j, j in enumerate(highest_trumps):
                self.can_follow_suit[index_i][index_j] = info_suits[index_i][index_j]

        return

    def legal_moves(self) -> set[Card]:
        hand = self.hands[self.current_player]

        leading_suit = self.tricks[-1].leading_suit()

        if leading_suit is None:
            return hand

        follow = set()
        trump = set()
        trump_higher = set()
        highest_trump_value = self.tricks[-1].highest_trump().order()
        for card in hand:
            if card.suit == leading_suit:
                follow.add(card)
            if card.suit == 0:
                trump.add(card)
                if card.order() > highest_trump_value:
                    trump_higher.add(card)

        # if follow:
        if follow and leading_suit != 0:
            return follow

        # current_winner = self.tricks[-1].winner()
        # if (current_winner+self.current_player) % 2 == 0:
        #     return hand

        return trump_higher or trump or hand

    def do_move(self, card: Card, mode: str = "normal"):
        """Play a card and update the game state"""
        if mode == "normal":
            self.update_possible_cards(card, False)
            if self.current_player == self.own_position:
                self.hands[self.current_player].remove(card)
        elif mode == "mcts_move":
            self.update_possible_cards(card, True)
            self.hands[self.current_player].remove(card)
        elif mode == "simulation":
            self.hands[self.current_player].remove(card)
        else:
            raise Exception("Invalid mode")

        self.tricks[-1].add_card(card)

        self.cards_left[self.current_player] -= 1
        if self.tricks[-1].trick_complete():
            winner = self.tricks[-1].winner()
            team_winner = team(winner)

            self.pit_check += team_winner

            points = self.tricks[-1].points()
            meld = self.tricks[-1].meld()

            self.points[team_winner] += points
            self.meld[team_winner] += meld

            if self.round_complete():

                # Winner of last trick gets 10 points
                self.points[team_winner] += 10
                defending_team = 1 - self.declaring_team

                # Check if the round has "pit"
                if self.pit_check == self.declaring_team * 8:
                    self.meld[self.declaring_team] += 100

                # Check if the round has "nat"
                if (
                    self.points[self.declaring_team] + self.meld[self.declaring_team]
                    <= self.points[defending_team] + self.meld[defending_team]
                ):
                    self.final_score[defending_team] = 162 + self.meld[defending_team] + self.meld[self.declaring_team]
                    self.final_score[self.declaring_team] = 0
                else:
                    self.final_score[0] = self.points[0] + self.meld[0]
                    self.final_score[1] = self.points[1] + self.meld[1]

            else:
                self.tricks.append(Trick(winner))
                self.current_player = winner
        else:
            self.current_player = (self.current_player + 1) % 4

    def undo_move(self, card: Card, reverse_possible_cards: bool = False):
        """Undo the last move made in the game. Can only be used for moves within mcts"""
        if self.tricks[-1].cards == [] or self.round_complete():

            if self.round_complete():

                # Reverse the: Check if the round has "nat"
                self.final_score[0] = 0
                self.final_score[1] = 0

                # Reverse the: Check if the round has "pit"
                if self.pit_check == self.declaring_team * 8:
                    self.meld[self.declaring_team] -= 100

                # Reverse the: Winner of last trick gets 10 points
                self.points[team(self.tricks[-1].winner())] -= 10
            else:
                self.tricks.pop()
                self.current_player = (self.tricks[-1].starting_player + 3) % 4

            team_winner = team(self.tricks[-1].winner())

            points = self.tricks[-1].points()
            meld = self.tricks[-1].meld()

            self.points[team_winner] -= points
            self.meld[team_winner] -= meld

            self.pit_check -= team_winner
        else:
            self.current_player = (self.current_player + 3) % 4

        self.cards_left[self.current_player] += 1

        self.tricks[-1].remove_card(card)

        self.hands[self.current_player].add(card)

        if reverse_possible_cards:
            for i in range(4):
                self.possible_cards[i] |= self.removed_cards[i]

    # determinized alternative representation with obtained cards split per team
    def to_nparray_determinized_alt(self):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 10), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()
        # Set the locations of the cards in the hands
        for index, cards in enumerate(self.hands):
            for card in cards:
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.own_position) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.own_position) % 4
            ] = 1

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                if trick.winner % 1: #TODO
                    card_location[8 * (card.id // 10) + card.id % 10][8] = 1
                else:
                    card_location[8 * (card.id // 10) + card.id % 10][9] = 1

        self.tijden[1] += time.time() - now
        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.own_position) % 4] = 1

        array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.own_position % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now
        return np.concatenate((card_location.flatten(), array))

    # normal regular one bartstyle
    def to_nparray(self):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 9), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()
        # Set the locations of the cards in the hands
        for index, cards in enumerate(self.possible_cards):
            for card in cards:
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.own_position) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.own_position) % 4
            ] = 1

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                card_location[8 * (card.id // 10) + card.id % 10][8] = 1

        self.tijden[1] += time.time() - now
        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.own_position) % 4] = 1

        array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.own_position % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now
        return np.concatenate((card_location.flatten(), array))

    # alt state representation, bart style iput
    def to_nparray_alt(self):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 10), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()
        # Set the locations of the cards in the hands
        for index, cards in enumerate(self.possible_cards):
            for card in cards:
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.own_position) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.own_position) % 4
            ] = 1

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                if trick.winner:
                    card_location[8 * (card.id // 10) + card.id % 10][8] = 1
                else:
                    card_location[8 * (card.id // 10) + card.id % 10][9] = 1

        self.tijden[1] += time.time() - now
        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.own_position) % 4] = 1

        array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.own_position % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now
        return np.concatenate((card_location.flatten(), array))

    #determinized
    def to_nparray_nope(self):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 9), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()
        # Set the locations of the cards in the hands
        for index, cards in enumerate(self.hands):
            for card in cards:
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.own_position) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.own_position) % 4
            ] = 1

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                card_location[8 * (card.id // 10) + card.id % 10][8] = 1

        self.tijden[1] += time.time() - now
        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.own_position) % 4] = 1

        array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.own_position % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now
        return np.concatenate((card_location.flatten(), array))

    #determinized with hands input
    def to_nparray_hand(self, determinization):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 9), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()

        #own_hand_as_id = [
        #    card_transform(card.id, ["k", "h", "r", "s"].index(round.trump_suit))
        #    for card in round.player_hands[self.own_position]
        #]

        # Set the locations of the cards in the hands
        for index, cards in enumerate(determinization):
            for _card in cards:
                card = Card(card_transform(_card.id, ["k", "h", "r", "s"].index(self._trump_suit)))
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.own_position) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.own_position) % 4
            ] = 1

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                card_location[8 * (card.id // 10) + card.id % 10][8] = 1

        self.tijden[1] += time.time() - now
        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.own_position) % 4] = 1

        array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.own_position % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now

        return np.concatenate((card_location.flatten(), array))

    #determinized with hands and alt array input from CURRENT PLAYER
    def to_nparray_hand_2(self, determinization):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 10), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()

        #own_hand_as_id = [
        #    card_transform(card.id, ["k", "h", "r", "s"].index(round.trump_suit))
        #    for card in round.player_hands[self.own_position]
        #]

        # Set the locations of the cards in the hands
        for index, cards in enumerate(determinization):
            for _card in cards:
                card = Card(card_transform(_card.id, ["k", "h", "r", "s"].index(self._trump_suit)))
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.current_player) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.current_player) % 4
            ] = 1

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                if trick.winner:
                    card_location[8 * (card.id // 10) + card.id % 10][8] = 1
                else:
                    card_location[8 * (card.id // 10) + card.id % 10][9] = 1

        self.tijden[1] += time.time() - now
        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.own_position) % 4] = 1

        array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.own_position % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now

        return np.concatenate((card_location.flatten(), array))

    # alt state representation, bart style input to nn, for other players != own_position
    def to_nparray_alt_op(self):
        """
        Convert the game state to a numpy array own position will become index 0
        first 32x9 array: 32 cards, 9 possible locations by one of the 4 players in one of the 4 centre positions or already played
        second 11 array: starting player 4, current player 4, declaring 1, points 2
        """
        card_location = np.zeros((32, 10), dtype=np.float16)  # 32 cards, 9 possible locations by one of the 4
        # players in one of the 4 centre positions or already played
        now = time.time()
        # Set the locations of the cards in the hands
        own_hand = self.hands[self.current_player]
        own_hand_as_id = [card.id for card in own_hand]
        
        local_team = team(self.current_player)
        own_position_team = team(self.own_position)
        original_hand = self.hands[self.own_position]
        all_players = [0,1,2,3]
        other_players = all_players.pop(self.current_player)

        played_cards = []
       
        # Set the locations of the cards in the centre
        for index, card in enumerate(self.tricks[-1].cards):
            # print("tricks", self.tricks[-1].cards)
            card_location[8 * (card.id // 10) + card.id % 10][
                4 + (self.tricks[-1].starting_player + index - self.current_player) % 4
            ] = 1
            played_cards.append(card.id)

        # Set the locations of the cards already played
        for trick in self.tricks[:-1]:
            for card in trick.cards:
                if own_position_team == local_team:
                    if trick.winner:
                        card_location[8 * (card.id // 10) + card.id % 10][8] = 1
                    else:
                        card_location[8 * (card.id // 10) + card.id % 10][9] = 1
                else:
                    if trick.winner:
                        card_location[8 * (card.id // 10) + card.id % 10][9] = 1
                    else:
                        card_location[8 * (card.id // 10) + card.id % 10][8] = 1
                played_cards.append(card.id)

        self.tijden[1] += time.time() - now
        now = time.time()


        not_own_hand_as_id = set([suit * 10 + value for suit in range(4) for value in range(8)]) - set(own_hand_as_id) - set(played_cards)
        possible_cards_ids = [set(id for id in not_own_hand_as_id) for _ in range(4)]


        local_possible_cards = [set([Card(id) for id in not_own_hand_as_id]) for _ in range(4)]

        # self.highest_trumps is the rank of potentially highest trump player could have
        for player in range(4):
            if player != self.current_player:
                to_remove_suits = []
                # start with assumption that each player has some trumps
                trumps = True
                for index, suit in enumerate(self.can_follow_suit[player]):
                    if not suit:
                        cards = [0,1,2,3,4,5,6,7]
                        suit_base = index * 10
                        to_remove_suits.extend([i + suit_base for i in cards])
                        if suit == 0: # Player cant follow trump suit
                            trumps = False
                to_remove_suits = set(to_remove_suits)
                
                if self.highest_trumps[player] != 15 and trumps:
                    #        7, 8,  9, 10,  J,  Q,  K,  A
                    #        0, 1,  2,  3,  4,  6,  7,  8
                    ranks = [8, 9, 14, 12, 15, 10, 11, 13]
                    to_remove_trumps = [i for i in ranks if i > self.highest_trumps[player]]
                    card_to_rank = dict(zip(ranks,[0,1,2,3,4,5,6,7]))
                    to_remove_trumps = set([card_to_rank.get(x, x) for x in to_remove_trumps])
                else:
                    to_remove_trumps = set()

                possible_cards_ids[player] = possible_cards_ids[player] - to_remove_suits - to_remove_trumps
                local_possible_cards[player] = set([Card(id) for id in possible_cards_ids[player]])
            else:
                local_possible_cards[player] = set([Card(id) for id in own_hand_as_id]) #deze?

        for index, cards in enumerate(local_possible_cards):
            for card in cards:
                card_location[8 * (card.id // 10) + card.id % 10][(index - self.current_player) % 4] = 1
        self.tijden[0] += time.time() - now

        now = time.time()

        card_location = np.where(
            np.logical_and(card_location, np.sum(card_location, axis=1, keepdims=True) > 1),
            card_location / np.sum(card_location, axis=1, keepdims=True),
            card_location,
        )

        self.tijden[2] += time.time() - now
        now = time.time()
        if not (np.sum(card_location, axis=1) == 1).all():
            print(card_location, flush=True)
            print(np.sum(card_location, axis=1))
            print(self.possible_cards, flush=True)
            for trick in self.tricks:
                print(trick.cards)
            raise ValueError("Some cards are not in the array")

        array = np.zeros(11, dtype=np.float16)

        array[(self.tricks[-1].starting_player - self.current_player) % 4] = 1

        #TODO werkt dit?
        array[4 + self.current_player % 4] = 1
        #array[4 + (self.current_player - self.own_position) % 4] = 1

        own_team = self.current_player % 2

        if self.declaring_team == own_team:
            array[8] = 1
        else:
            array[8] = 0

        # Set the points
        if self.round_complete():
            array[9] = self.final_score[own_team] / 100
            array[10] = self.final_score[1 - own_team] / 100
        else:
            array[9] = (self.points[own_team] + self.meld[own_team]) / 100
            array[10] = (self.points[1 - own_team] + self.meld[1 - own_team]) / 100
        self.tijden[3] += time.time() - now
        return np.concatenate((card_location.flatten(), array))

    def round_complete(self) -> bool:
        if len(self.hands[self.own_position]) == 0 and self.tricks[-1].trick_complete():
            return True
        return False

    def get_score(self, player: int) -> int:
        if self.final_score[0] == 0 and self.final_score[1] == 0:
            print("=======================Game not finished yet=======================")
        local_team = team(player)
        return self.final_score[local_team] - self.final_score[1 - local_team]

    def get_prediction_score(self, player: int, declarer: int, trump_suit):
        if self.final_score[0] == 0 and self.final_score[1] == 0:
            print("=======================Game not finished yet=======================")
        local_team = team(player)
        prediction_score = np.zeros(5)
        options = ["k","h","r","s","p"]

        if player == declarer:
            if self.final_score[local_team] > self.final_score[local_team ^ 1]:
                prediction_score[options.index(trump_suit)] = 1
            else:
                prediction_score[4] = 1 # Should have passed
            
        elif local_team == team(declarer): #teammate of declarer
            prediction_score[4] = 1 # pass was right option?
        else: #player is team that did not declarer
            prediction_score[4] = 1 # pass was right option?

        return prediction_score