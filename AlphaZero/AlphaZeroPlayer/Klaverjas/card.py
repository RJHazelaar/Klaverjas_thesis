from __future__ import annotations  # To use the class name in the type hinting


class Card:
    """AlphaZero card class"""

    def __init__(self, id: int):
        self.id = id
        self.suit = id // 10
        self.value = id % 10

    def __repr__(self):
        return str(self.id)

    def __eq__(self, other: Card):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def order(self) -> int:
        """Returns the rank of the card"""
        if self.suit == 0:
            return [8, 9, 14, 12, 15, 10, 11, 13][self.value]
        return [0, 1, 2, 6, 3, 4, 5, 7][self.value]

    def points(self) -> int:
        """Returns the value of the card"""
        if self.suit == 0:
            return [0, 0, 14, 10, 20, 3, 4, 11][self.value]
        return [0, 0, 0, 10, 2, 3, 4, 11][self.value]

    def get_highest_card(self, suit):
        return self.cardsleft[['k', 'h', 'r', 's'].index(suit)][-1] # Returns highest card from each suit
    