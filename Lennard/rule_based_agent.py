class Rule_player:
    #Returns the card for the rule-based player
    def get_card_good_player(self, round, player):
        # player = round.current_player
        trick = round.tricks[-1]
        trump = round.trump_suit
        legal_moves = round.legal_moves()
        cant_follow = 0
        if len(trick.cards) == 0:
            for card in legal_moves:
                if card.value == round.get_highest_card(card.suit):
                    return card
            return self.get_lowest_card(legal_moves, trump)

        if legal_moves[0].suit == trick.cards[0].suit:
            cant_follow = 1
            
        if len(trick.cards) == 1:
            for card in legal_moves:
                if card.value == round.get_highest_card(trick.cards[0].suit):
                    return card
            return self.get_lowest_card(legal_moves, trump)

        if len(trick.cards) == 2:
            if cant_follow:
                return self.get_lowest_card(legal_moves, trump)

            if trick.cards[0].value == round.get_highest_card(trick.cards[0].suit):
                return self.get_highest_card(legal_moves, trump)


            for card in legal_moves:
                if card.value == round.get_highest_card(trick.cards[0].suit):
                    return card

            return self.get_lowest_card(legal_moves, trump)

        else:
            
            if trick.winner(trump) %2 == round.current_player %2:
                return self.get_highest_card(legal_moves, trump)

            highest = trick.highest_card(trump)
            for card in legal_moves:
                if card.order(trump) > highest.order(trump):
                    return card

            return self.get_lowest_card(legal_moves, trump)
    
    def get_lowest_card(self, legal_moves, trump):
        lowest_points = 21
        for card in legal_moves:
            if card.points(trump) < lowest_points:
                lowest_card = card
                lowest_points = card.points(trump)
        return lowest_card

    def get_highest_card(self, legal_moves, trump):
        highest_points = -1
        for card in legal_moves:
            if card.points(trump) > highest_points:
                highest_card = card
                highest_points = card.points(trump)
        return highest_card

