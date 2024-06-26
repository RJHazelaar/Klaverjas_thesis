from copy import deepcopy

def card_to_suit(card: int) -> int:
    return card // 10


def card_to_value(card: int) -> int:
    return card % 10


def team(player: int) -> int:
    return player % 2


def card_transform(card: int, trump_suit: int) -> int:
    "Makes cards of the trump suit to have suit 0 and cards with suit 0 have suit trump_suit"
    suit = card_to_suit(card)
    if suit == trump_suit:
        return card_to_value(card)
    elif suit == 0:
        return trump_suit * 10 + card_to_value(card)
    else:
        return card


def card_untransform(card: int, trump_suit: int) -> int:
    "Makes cards of the trump suit to have suit trump_suit and cards with suit trump_suit have suit 0"
    suit = card_to_suit(card)
    if suit == 0:
        return trump_suit * 10 + card_to_value(card)
    elif suit == trump_suit:
        return card_to_value(card)
    else:
        return card

def hand_transform(_player_hands: list, trump_suit: int) -> list:
    player_hands = deepcopy(_player_hands)
    for hand in player_hands:
        for index, card in enumerate(hand):
            hand[index]= card_transform(card.id, trump_suit)
    return player_hands