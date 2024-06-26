from Lennard.tricks import Trick
from Lennard.deck import Deck, Card
import numpy as np

def team(player):
    return player % 2

def other_team(player):
    return (player + 1) % 2

class Round:
    def __init__(self, starting_player, trump_suit, declarer, model=None, **kwargs):
        self.starting_player = starting_player
        self.current_player = starting_player
        self.declarer = declarer
        self.bidders = [0,0,0,0] # Player that did a bid
        self.passes = 0
        if model is not None:
            options = ["k","h","r","s","p"]
            self.trump_suit = "k"
            declarer = starting_player
            self.tricks = [Trick(starting_player)]
            self.points = [0,0]
            self.meld = [0, 0]

            self.cardsleft = [[i for i in range(7,15)] for j in range(4)]
            for i in range(4):
                if ["k", "h", "r", "s"][i] == self.trump_suit:
                    order = [8, 9, 14, 12, 15, 10, 11, 13]
                else:
                    order = [0, 1, 2, 6, 3, 4, 5, 7]
                ordered_list = [i for _, i in sorted(zip(order, self.cardsleft[i]))]
                self.cardsleft[i] = ordered_list
                
            self.deal()               

            # NEURAL NETWORK
            # TODO Vectorize to output all players simulteneously
            bidding_order = list(range(declarer, 4)) + list(range(0, declarer))
            for bidder in bidding_order:
                input_vector = self.hand_to_input_vector(bidder, starting_player)
                output = model(input_vector)
                self.bidders[bidder] = 1
                possible_trump_suit = options[np.argmax(output)] 
                if possible_trump_suit != "p":
                    self.declarer = bidder
                    self.trump_suit = possible_trump_suit
                    break
                else:
                    self.passes += 1
            
            if self.trump_suit == "p": # First declarer forced to make a decision != passing
                output = model(input_vector)[:-1]
                self.trump_suit = options[np.argmax(output)] #TODO Just use the previous output
            # NEURAL NETWORK

            self.declaring_team = team(self.declarer)

        else:    
            self.trump_suit = trump_suit
            self.declaring_team = team(self.declarer)
            self.tricks = [Trick(starting_player)]
            self.points = [0,0]
            self.meld = [0, 0]

            self.cardsleft = [[i for i in range(7,15)] for j in range(4)]
            for i in range(4):
                if ["k", "h", "r", "s"][i] == self.trump_suit:
                    order = [8, 9, 14, 12, 15, 10, 11, 13]
                else:
                    order = [0, 1, 2, 6, 3, 4, 5, 7]
                ordered_list = [i for _, i in sorted(zip(order, self.cardsleft[i]))]
                self.cardsleft[i] = ordered_list       

            self.deal()
    
    def round_in_progress(self, starting_player, trump_suit, declarer, cards_players):
        1 == 1

    def hand_to_input_vector(self, declarer, starting_player):
        all_cards = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]
        input_vector = np.in1d(all_cards, self.player_hands[declarer]).astype(int)
        position = np.array([i%4 for i in range(starting_player,starting_player+4)])
        position = np.where(position == declarer, 1, 0)
        return np.concatenate((input_vector, position))[np.newaxis]

    #Gives each player 8 cards to play with
    def deal(self):
        deck = Deck()
        deck.shuffle()
        self.player_hands = [deck.cards[0:8], deck.cards[8:16], deck.cards[16:24], deck.cards[24:32]]
        
    def set_cards(self, cards: list[str]):
        # self.player_hands = cards
        self.player_hands = [[], [], [], []]
        for index, hand in enumerate(cards):
            hand = hand.split()
            for card in hand:
                # print(type(card[0]), type(card[1:]))
                self.player_hands[index].append(Card(int(card[1:]), card[0]))
        
    #Returns the legal moves a player could make based on the current hand and played cards
    def legal_moves(self, player=None):
        if player is None:
            player = self.current_player
        hand = self.player_hands[player]
        trick = self.tricks[-1]
        leading_suit = trick.leading_suit()

        if leading_suit is None:
            return hand

        follow = []
        trump = []
        trump_higher = []
        highest_trump_value = trick.highest_trump(self.trump_suit).order(self.trump_suit)
        for card in hand:
            if card.suit == leading_suit:
                follow.append(card)
            if card.suit == self.trump_suit:
                trump.append(card)
                if card.order(self.trump_suit) > highest_trump_value:
                    trump_higher.append(card)

        if follow and leading_suit != self.trump_suit:
        # if follow:
            return follow

        # current_winner = trick.winner(self.trump_suit)
        # if (current_winner + player) % 2 == 0:
        #     return hand

        return trump_higher or trump or hand   
    
    #Checks whether the round is complete
    def is_complete(self):
        return len(self.tricks) == 8 and self.tricks[-1].is_complete()

    #Plays the card in a trick
    def play_card(self, card, player=None, check=True):

        if check and card not in self.legal_moves():
            print("HIER", card.value, card.suit)
            print([(card.value, card.suit) for card in self.legal_moves()])
            raise Exception("Illegal move")
        if player is None:
            player = self.current_player
        if player != self.current_player:
            raise Exception("Not your turn")
        self.tricks[-1].add_card(card)
        # print("hier1", card.value, card.suit)
        # for card in self.player_cards[player]:
        #     print("hier2", card.value, card.suit)
        self.player_hands[player].remove(card)
        if self.tricks[-1].is_complete():
            self.complete_trick()
        else:
            self.current_player = (self.current_player + 1) % 4
        
    
    #Checks whether the trick is complete and handles all variables
    def complete_trick(self):
        trick = self.tricks[-1]
        if trick.is_complete():
            for card in trick.cards:
                self.cardsleft[['k', 'h', 'r', 's'].index(card.suit)].remove(card.value)
            winner = trick.winner(self.trump_suit)
            points = trick.points(self.trump_suit)
            meld = trick.meld(self.trump_suit)
            self.points[team(winner)] += points
            self.meld[team(winner)] += meld

            if len(self.tricks) == 8:
                self.points[team(winner)] += 10
                defending_team = 1 - self.declaring_team
                
                if (self.points[self.declaring_team] + self.meld[self.declaring_team] <=
                        self.points[defending_team] + self.meld[defending_team]):
                    self.points[defending_team] = 162
                    self.meld[defending_team] += self.meld[self.declaring_team]
                    self.points[self.declaring_team] = 0
                    self.meld[self.declaring_team] = 0
                elif self.is_pit():
                    self.meld[self.declaring_team] += 100
            else:
                self.tricks.append(Trick(winner))
                self.current_player = winner
            return True
        return False

    #Checks whether all tricks are won by one team
    def is_pit(self):
        for trick in self.tricks:
            if team(self.declaring_team) != team(trick.winner(self.trump_suit)):
                return False
        return True

    def get_highest_card(self, suit):
        return self.cardsleft[['k', 'h', 'r', 's'].index(suit)][-1]
    
    def get_score(self, player):
        local_team = team(player)
        return self.points[local_team] - self.points[1-local_team] + self.meld[local_team] - self.meld[1-local_team]
        

