"""
Created 13 July 2020
author: Mckay Jensen

Teaching a computer to play Rook card game with adversarial neural networks
"""

import random
import torch
import torch.nn as nn


LR = 0.05  # the learning rate
VERBOSE = False

def printv(text):
    if VERBOSE:
        print(text)


class Card:

    def __init__(self, color, number):
        self.color = color
        assert self.color in ('red', 'yellow', 'green', 'black')
        self.number = number
        assert self.number in range(1, 15)
        if self.number == 5:
            self.points = 5
        elif self.number == 10 or self.number == 13:
            self.points = 10
        elif self.number == 1:
            self.points = 15
        else:
            self.points = 0


class Rook(Card):

    def __init__(self):
        self.color = None
        self.number = 0
        self.points = 20
        

def color_to_one_hot(color):
    """Returns a 4-d one-hot vector representing one of the four colors
    """
    assert color in ('red', 'yellow', 'green', 'black')
    out_vec = torch.zeros(4)
    i = 0 if color == 'red' else 1 if color == 'yellow' else 2 if color == 'green' else 3
    out_vec[i] = 1
    return out_vec


def vectorize_cards(cards):
    """Takes a list of cards and outputs a torch one-hot tensor representing those cards
    
    Order is Rook, 1-14 red, 1-14 yellow, 1-14 green, 1-14 black
    
    output out_vec is a 57-dimensional torch tensor of 0s and 1s
    """
    out_vec = torch.zeros(57)
    for card in cards:
        if isinstance(card, Rook):
            out_vec[0] = 1
        else:
            if card.color == 'red':
                offset = 0
            elif card.color == 'yellow':
                offset = 14
            elif card.color == 'green':
                offset = 28
            else:
                offset = 42
            out_vec[offset + card.number] = 1
    return out_vec


def card_from_index(index):
    """Takes an int from 0 to 56 and finds the corresponding card based on the same encoding used in vectorize_cards()
    """
    assert isinstance(index, int) and index >= 0 and index < 57
    if index == 0:
        return Rook()
    else:
        color_num = (index - 1) // 14
        color = 'red' if color_num == 0 else 'yellow' if color_num == 1 else 'green' if color_num == 2 else 'black'
        number = (index - 1) % 14 + 1
        return Card(color, number)

        

class ChooseCard(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 57)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.zero_grad()
    
    def get_zeros(self):
        return [torch.zeros_like(p) for p in self.parameters()]
    
    def get_grads(self):
        try:
            return [p.grad.detach().clone() for p in self.parameters()]
        except AttributeError:
            # gradients do not exist, return tensors of zeros instead
            return self.get_zeros()
    
    def forward(self, input_features):
        x = self.relu(self.l1(input_features))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        return x
    


class Game:

    def __init__(self):
        # set up the game state to be ready for the start of the game
        self.reset()
        # set up card choice neural nets
        self.firstPlayerChooseCard = ChooseCard(118)  # input dim is 4 + 2*57
        self.grad1 = self.firstPlayerChooseCard.get_grads()
        self.secondPlayerChooseCard = ChooseCard(175)  # input dim is 4 + 3*57
        self.grad2 = self.secondPlayerChooseCard.get_grads()
        self.thirdPlayerChooseCard = ChooseCard(175)
        self.grad3 = self.thirdPlayerChooseCard.get_grads()
        self.fourthPlayerChooseCard = ChooseCard(118)
        self.grad4 = self.fourthPlayerChooseCard.get_grads()
    
    def reset(self):
        # create list of all cards in the deck
        cards = [Card(color, number) for color in ['red', 'yellow', 'green', 'black'] for number in range(1, 15)] \
                + [Rook()]
        # shuffle deck and deal
        random.shuffle(cards)
        self.players = [cards[13*i:13*(i+1)] for i in range(4)]
        self.widow = cards[13*4:]
        # do bidding
        self.bid_winner, self.bid_amount = self.bid()
        # take widow and choose trump color
        self.trump = self.choose_trump()
        self.trump_vec = color_to_one_hot(self.trump)
        self.score0 = 0  # score for players 0 and 2 (team 0)
        self.score1 = 0  # score for players 1 and 3 (team 1)
        self.cards_balance = 0  # difference of cards taken by team 0 and by team 1

    def bid(self):
        # TODO: Use ML for bidding
        # for now just always give to player 0 for random amount
        bid_winner = 0
        bid_amount = random.choice([100 + x*5 for x in range(21)])
        return bid_winner, bid_amount

    def choose_trump(self):
        # TODO: Use ML for choosing trump
        # use a simple strategy for now: choose trump as color with most cards
        # put into widow lowest cards of non-trump colors
        all_cards = self.players[self.bid_winner] + self.widow
        colors = [card.color for card in self.players[self.bid_winner]]
        trump = max(set(colors), key=colors.count)
        not_trump_indices = [i for i, card in enumerate(all_cards) if card.color != trump]
        to_widow = []
        for i in range(2, 15):
            for j in not_trump_indices:
                if all_cards[j].number == i:
                    to_widow.append(j)
                if len(to_widow) == 5:
                    break
            if len(to_widow) == 5:
                break
        self.players[self.bid_winner] = [card for i, card in enumerate(all_cards) if i not in to_widow]
        self.widow = [card for i, card in enumerate(all_cards) if i in to_widow]
        return trump

    def resolve_hand(self, cards, player_order):
        suite = cards[0].color
        highest_number = cards[0].number
        winner = 0
        for i in range(1, 4):
            if suite != self.trump and (cards[i].color == self.trump or isinstance(cards[i], Rook)):
                suite = self.trump
                highest_number = cards[i].number
                winner = i
            elif cards[i].color == suite and ((cards[i].number > highest_number and highest_number != 1) \
                                              or cards[i].number == 1):
                highest_number = cards[i].number
                winner = i
        points = sum(card.points for card in cards)
        return player_order[winner], points
    
    def get_valid_cards(self, first_card, cards_in_hand):
        """Determines which cards in a player's hand can be legally played given the first card played in a round.
        first card is a Card object.
        cards_in_hand is a list of Card objects.
        returns a 57d one-hot vector representing legal plays.
        """
        same_color_cards = [card for card in cards_in_hand if \
                            card.color == first_card.color or \
                            (isinstance(card, Rook) and first_card.color == self.trump)]
        if same_color_cards:
            return vectorize_cards(same_color_cards)
        trump_cards = [card for card in cards_in_hand if card.color == self.trump or isinstance(card, Rook)]
        if trump_cards:
            return vectorize_cards(trump_cards)
        return vectorize_cards(cards_in_hand)
        
    def choose_card(self, cards, player_order):
        num_cards_down = len(cards)
        player = player_order[num_cards_down]
        # Have a separate neural network for each num_cards_down
        # Features to include:
            # cards in hand
            # cards already played
            # cards that can be played still
            # what color is trump
        cards_in_hand = vectorize_cards(self.players[player])
        if num_cards_down == 0:
            possible_plays = vectorize_cards(sum(
                        (self.players[p] for p in player_order[1:]), []
                    ))
            if player != self.bid_winner:
                possible_plays += vectorize_cards(self.widow)
            input_features = torch.cat((self.trump_vec, cards_in_hand, possible_plays))
            card_choice_vec = self.firstPlayerChooseCard(input_features)
        elif num_cards_down == 1:
            possible_plays = vectorize_cards(sum(
                        (self.players[p] for p in player_order[2:]), []
                    ))
            if player != self.bid_winner:
                possible_plays += vectorize_cards(self.widow)
            cards_down = vectorize_cards(cards)
            input_features = torch.cat((self.trump_vec, cards_in_hand, possible_plays, cards_down))
            card_choice_vec = self.secondPlayerChooseCard(input_features)
        elif num_cards_down == 2:
            possible_plays = vectorize_cards(self.players[player_order[3]])
            if player != self.bid_winner:
                possible_plays += vectorize_cards(self.widow)
            cards_down = vectorize_cards(cards)
            input_features = torch.cat((self.trump_vec, cards_in_hand, possible_plays, cards_down))
            card_choice_vec = self.thirdPlayerChooseCard(input_features)
        elif num_cards_down == 3:
            cards_down = vectorize_cards(cards)
            input_features = torch.cat((self.trump_vec, cards_in_hand, cards_down))
            card_choice_vec = self.fourthPlayerChooseCard(input_features)
        # determine the best available card from card_choice_vec
        available_plays = card_choice_vec.detach().clone()
        if num_cards_down == 0:
            available_plays *= cards_in_hand
        else:
            available_plays *= self.get_valid_cards(cards[0], self.players[player])
        best_card_index = available_plays.argmax().item()
        # update the gradients determining that card
        card_choice_vec[best_card_index].backward()
        if num_cards_down == 0:
            new_grads = self.firstPlayerChooseCard.get_grads()
            for i, g in enumerate(self.grad1):
                g += new_grads[i]
            self.firstPlayerChooseCard.zero_grad()
        elif num_cards_down == 1:
            new_grads = self.secondPlayerChooseCard.get_grads()
            for i, g in enumerate(self.grad2):
                g += new_grads[i]
            self.secondPlayerChooseCard.zero_grad()
        elif num_cards_down == 2:
            new_grads = self.thirdPlayerChooseCard.get_grads()
            for i, g in enumerate(self.grad3):
                g += new_grads[i]
            self.thirdPlayerChooseCard.zero_grad()
        elif num_cards_down == 3:
            new_grads = self.fourthPlayerChooseCard.get_grads()
            for i, g in enumerate(self.grad4):
                g += new_grads[i]
            self.fourthPlayerChooseCard.zero_grad()
        # interpret the best card index as a Card and update the game state
        choice_dict = card_from_index(best_card_index).__dict__
        for i, card in enumerate(self.players[player]):
            if card.__dict__ == choice_dict:
                return self.players[player].pop(i)
        assert False, f'shouldn\'t arrive here! player is {player} and card is {choice_dict}'

    def play_hand(self, starting_player):
        player_order = [(i+starting_player) % 4 for i in range(4)]
        cards = []
        for player in player_order:
            cards.append(self.choose_card(cards, player_order))
            printv(f'player {player} plays rook' if isinstance(cards[-1], Rook) else \
                   f'player {player} plays {cards[-1].color} {cards[-1].number}')
        winner, points = self.resolve_hand(cards, player_order)
        if winner % 2 == 0:
            self.score0 += points
        else:
            self.score1 += points
        return winner

    def play(self, train=False):
        printv(f'Player {self.bid_winner} won the bid for {self.bid_amount}.')
        printv(f'{self.trump} is trump')
        starting_player = self.bid_winner
        for i in range(13):
            printv(f'Round {i+1}:')
            starting_player = self.play_hand(starting_player)
        # take points from widow
        if starting_player % 2 == 0:
            self.score0 += sum(card.points for card in self.widow)
            self.cards_balance += 5
        else:
            self.score1 += sum(card.points for card in self.widow)
            self.cards_balance -= 5
        if self.cards_balance > 0:
            self.score0 += 20
        else:
            self.score1 += 20
        # subtract bid amount if the bid winner goes set
        if self.bid_winner == 0 or self.bid_winner == 2 and self.score0 < self.bid_amount:
            self.score0 -= self.bid_amount
        elif self.bid_winner == 1 or self.bid_winner == 3 and self.score1 < self.bid_amount:
            self.score1 -= self.bid_amount
        printv('Even team wins.' if self.score0 > self.score1 else \
               'Odd team wins.' if self.score0 < self.score1 else \
               'It\'s a tie.')
        if train:
            # update neural net parameters based on acccumulated gradients & who won the game
            # if a player won, reinforce the params based on the grad, otherwise do opposite
            sign = 1 if self.score0 > self.score1 else -1 if self.score0 < self.score1 else 0
            for p, grad in zip(self.firstPlayerChooseCard.parameters(), self.grad1):
                p.data += sign * LR * grad
            self.grad1 = self.firstPlayerChooseCard.get_zeros()
            for p, grad in zip(self.secondPlayerChooseCard.parameters(), self.grad2):
                p.data -= sign * LR * grad
            self.grad2 = self.secondPlayerChooseCard.get_zeros()
            for p, grad in zip(self.thirdPlayerChooseCard.parameters(), self.grad3):
                p.data += sign * LR * grad
            self.grad3 = self.thirdPlayerChooseCard.get_zeros()
            for p, grad in zip(self.fourthPlayerChooseCard.parameters(), self.grad4):
                p.data -= sign * LR * grad
            self.grad4 = self.fourthPlayerChooseCard.get_zeros()
        # reset the game state to play again
        self.reset()
    
    def train(self, rounds=100):
        for i in range(rounds):
            self.play(train=True)
            if (i+1) % 100 == 0 or (i+1) == rounds:
                print(f'Completed {i+1} / {rounds} training rounds.')



game = Game()
game.train()
