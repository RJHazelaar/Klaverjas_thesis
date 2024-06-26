from Lennard.deck import Deck, Card
from Lennard.helper import card_to_suit, card_to_value

class Trick:
    def __init__(self, starting_player):
        self.cards = []
        self.starting_player = starting_player

    #Adds the played card to itself
    def add_card(self, card):
        self.cards.append(card)

    #Checks whether the trick is over
    def is_complete(self):
        return len(self.cards) == 4

    #Returns the leading suit of the trick
    def leading_suit(self):
        if self.cards:
            return self.cards[0].suit

    #Returns the winner of the trick
    def winner(self, trump_suit):
        highest = self.cards[0] 
        for card in self.cards:
            if (card.order(trump_suit) > highest.order(trump_suit) and
                (card.suit == self.leading_suit() or
                 card.suit == trump_suit)):
                highest = card        
        return (self.starting_player + self.cards.index(highest)) % 4

    #Returns the highest card currently played in this trick
    def highest_card(self, trump_suit):
        highest = self.cards[0] 
        for card in self.cards:
            if (card.order(trump_suit) > highest.order(trump_suit) and
                (card.suit == self.leading_suit() or
                 card.suit == trump_suit)):
                highest = card 
        return highest

    #Returns the player that is currently at turn
    def to_play(self):
        return (self.starting_player + len(self.cards)) % 4

    #Returns the total points of the played cards in this trick
    def points(self, trump_suit):
        return sum(card.points(trump_suit) for card in self.cards)
    
    #Returns the highest played trump card
    def highest_trump(self, trump_suit):
        return max(self.cards,
                   default=Card(7, trump_suit).order(trump_suit),
                   key=lambda card: card.order(trump_suit))
    
    #Returns the meld points in this trick
    def meld(self, trump_suit: int):
        trump_suit = ['k', 'h', 'r', 's'].index(trump_suit)
        cards = [card.id for card in self.cards]
        values = [card_to_value(card) for card in cards]
        sorted = cards.copy()
        sorted.sort()
        point = 0

        # King and Queen of trump suit
        if trump_suit*10 + 5 in cards and trump_suit*10 + 6 in cards:
            point += 20

        # four consecutive cards of the same suit
        if card_to_suit(sorted[0]) == card_to_suit(sorted[3]) and card_to_value(sorted[0]) == card_to_value(sorted[3]) - 3:
            return point + 50

        # three consecutive cards of the same suit
        if ((card_to_suit(sorted[0]) == card_to_suit(sorted[2]) and card_to_value(sorted[0]) == card_to_value(sorted[2]) - 2) or
            (card_to_suit(sorted[1]) == card_to_suit(sorted[3]) and card_to_value(sorted[1]) == card_to_value(sorted[3]) - 2)):
            return point + 20
        
        # four cards of value Jack
        if len(set(values)) == 1 and values[0] == 4:
            return 200

        # four cards of the same face value
        if len(set(values)) == 1:
            return 100
        
        return point
    
    
