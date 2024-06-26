from AlphaZero.AlphaZeroPlayer.Klaverjas.card import Card
from AlphaZero.AlphaZeroPlayer.Klaverjas.helper import card_to_suit, card_to_value


class Trick:
    """AlphaZero trick class"""

    def __init__(self, starting_player: int):
        self.cards = []
        self.starting_player = starting_player

    def __repr__(self):
        return str(self.cards)

    def __eq__(self, other):
        # raise NotImplementedError
        return self.cards == other.cards and self.starting_player == other.starting_player

    def __hash__(self):
        raise NotImplementedError
        return hash(tuple(self.cards))

    def add_card(self, card: Card):
        """Adds the played card to itself"""
        self.cards.append(card)

    def remove_card(self, card: Card):
        """Removes the played card from itself"""

        if self.cards.pop() != card:
            raise ValueError("Card not in trick")

    def trick_complete(self) -> bool:
        """Checks whether the trick is over"""
        return len(self.cards) == 4

    def leading_suit(self) -> int:
        """Returns the leading suit of the trick"""
        if self.cards:
            return self.cards[0].suit

    def winner(self) -> int:
        """Returns the winner of the trick"""
        highest = self.cards[0]
        for card in self.cards:
            if card.order() > highest.order() and (card.suit == self.leading_suit() or card.suit == 0):
                highest = card
        return (self.starting_player + self.cards.index(highest)) % 4

    def highest_card(self) -> Card:
        """Returns the highest card currently played in this trick"""
        highest = self.cards[0]
        for card in self.cards:
            if card.order() > highest.order() and (card.suit == self.leading_suit() or card.suit == 0):
                highest = card
        return highest

    def to_play(self) -> int:
        """Returns the player that is currently at turn"""
        return (self.starting_player + len(self.cards)) % 4

    def points(self) -> int:
        """Returns the total points of the played cards in this trick"""
        return sum(card.points() for card in self.cards)

    def highest_trump(self) -> Card:
        """Returns the highest played trump card"""
        return max(self.cards, default=Card(0).order(), key=lambda card: card.order())

    def meld(self) -> int:
        """Returns the meld points in this trick"""
        cards = [card.id for card in self.cards]
        values = [card.value for card in self.cards]
        sorted = cards.copy()
        sorted.sort()
        point = 0

        # King and Queen of trump suit
        if 5 in cards and 6 in cards:
            point += 20

        # four consecutive cards of the same suit
        if (
            card_to_suit(sorted[0]) == card_to_suit(sorted[3])
            and card_to_value(sorted[0]) == card_to_value(sorted[3]) - 3
        ):
            return point + 50

        # three consecutive cards of the same suit
        if (
            card_to_suit(sorted[0]) == card_to_suit(sorted[2])
            and card_to_value(sorted[0]) == card_to_value(sorted[2]) - 2
        ) or (
            card_to_suit(sorted[1]) == card_to_suit(sorted[3])
            and card_to_value(sorted[1]) == card_to_value(sorted[3]) - 2
        ):
            return point + 20

        # four cards of value Jack
        if len(set(values)) == 1 and values[0] == 4:
            return 200

        # four cards of the same face value
        if len(set(values)) == 1:
            return 100

        return point
