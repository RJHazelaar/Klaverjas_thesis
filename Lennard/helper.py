suits_list = ["Hearth", "Diamond", "Spade", "Club"]
values_list = ["7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]

# def id_to_suit(id: int) -> str:
#     return suits_list[id // 10]

# def id_to_value(id: int) -> str:
#     return values_list[id % 10]

# def strings_to_id(value: str, suit: str) -> int:
#     return suits_list.index(suit) * 10 + values_list.index(value)

# def values_to_id(value: str) -> int:
#     return values_list.index(value)

# def suits_to_id(suit: str) -> int:
#     return suits_list.index(suit) * 10

# def team(player):
#     return player % 2

# def other_team(player):
#     return (player + 1) % 2

def card_to_suit(card: int) -> int:
    return card // 10

def card_to_value(card: int) -> int:
    return card % 10

# def card_to_order2(card: int, trump: int) -> int:
#     if card // 10 == trump:
#         return [8, 9, 14, 12, 15, 10, 11, 13][card % 10]
#     return [0, 1, 2, 6, 3, 4, 5, 7][card % 10]

# def card_to_points2(card: int, trump: int) -> int:
#     if card // 10 == trump:
#         return [0, 0, 14, 10, 20, 3, 4, 11][card % 10]
#     return [0, 0, 0, 10, 2, 3, 4, 11][card % 10]

# def card_to_order(card: int) -> int:
#     if card // 10 == 0:
#         return [8, 9, 14, 12, 15, 10, 11, 13][card % 10]
#     return [0, 1, 2, 6, 3, 4, 5, 7][card % 10]

# def card_to_points(card: int) -> int:
#     if card // 10 == 0:
#         return [0, 0, 14, 10, 20, 3, 4, 11][card % 10]
#     return [0, 0, 0, 10, 2, 3, 4, 11][card % 10]

# def print_moves(moves):
#     print(list(map(card_to_string, moves)))

# def card_to_string(card):
#     return values_list[card_to_value(card)] + suits_list[card_to_suit(card)]

# def print_hands(hands):
#     print("Hands:")
#     print(list(map(card_to_string, hands[0])))
#     print(list(map(card_to_string, hands[1])))
#     print(list(map(card_to_string, hands[2])))
#     print(list(map(card_to_string, hands[3])))
    
# def cards_to_suit(cards: list) -> int:
#     return [card_to_suit(card) for card in cards]

# def cards_to_value(cards: list) -> int:
#     return [card_to_value(card) for card in cards]