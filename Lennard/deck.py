import random

class Card:
    def __init__(self, value, suit):
        """suit + value are ints"""
        self.value = value
        self.suit = suit
        self.id = ['k', 'h', 'r', 's'].index(suit)*10 + value-7
        
    def __repr__(self):
        return str(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __eq__(self, other: int):
        return self.id == other

    #Returns the rank of the card given the trump suit
    def order(self, trump_suit):
        if self.suit == trump_suit:
            return [8, 9, 14, 12, 15, 10, 11, 13][self.value - 7]
        return [0, 1, 2, 6, 3, 4, 5, 7][self.value - 7]

    #Returns the value of the card given the trump suit
    def points(self, trump_suit):
        if self.suit == trump_suit:
            return [0, 0, 14, 10, 20, 3, 4, 11][self.value - 7]
        return [0, 0, 0, 10, 2, 3, 4, 11][self.value - 7]

class Deck:
    def __init__(self):
        values = [7, 8, 9, 10, 11, 12, 13, 14]
        suits = ['k', 'h', 'r', 's']
        self.cards = []
        for value in values:
            for suit in suits:
                self.cards.append(Card(value, suit))
        
    def shuffle(self):
        random.shuffle(self.cards)