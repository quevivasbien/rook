"""
Created 13 July 2020
author: Mckay Jensen

Teaching a computer to play Rook card game with adversarial neural networks
"""

import random
import torch
import torch.nn as nn
import pickle

from math import exp
from time import time


MAX_LR = 0.10  # the learning rate
MIN_LR = 0.001
DECAY_RATE = 0.001  # exponential decay for learning rate, set to 0 for no decay
# parameters for adam optimization
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-6

PARAMS = 'params.pkl'



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



class ChoiceNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.cached_grads = None
        self.t = 0
    
    def get_zeros(self):
        return [torch.zeros_like(p) for p in self.parameters()]
    
    def get_grads(self):
        try:
            return [p.grad.detach().clone() for p in self.parameters()]
        except AttributeError:
            # gradients do not exist, return tensors of zeros instead
            return self.get_zeros()
    
    def get_cached_grads(self):
        if self.cached_grads is None:
            self.cached_grads = self.get_zeros()
        return self.cached_grads
    
    def cache_grads(self):
        new_grads = self.get_grads()
        for i, g in enumerate(self.get_cached_grads()):
            g += new_grads[i]
        self.zero_grad()
    
    def adam_update(self, step_size=-1):
        if self.t == 0:
            self.m = self.get_zeros()
            self.v = self.get_zeros()
        self.t += 1
        for m, v, g, p in zip(self.m, self.v, self.get_cached_grads(), self.parameters()):
            m = BETA1 * m + (1 - BETA1) * g
            mhat = m / (1 - BETA1**self.t)
            v = BETA2 * v + (1 - BETA2) * torch.pow(g, 2)
            vhat = v / (1 - BETA2)**self.t
            update = step_size * mhat / (torch.sqrt(vhat) + EPSILON)
            # check for nans before updating
            if not torch.isnan(update).any():
                p.data += update
        # reset cached gradients
        self.cached_grads = self.get_zeros()

    def forward(self, input_features):
        raise NotImplementedError
        
        
        
class ChooseBid(ChoiceNN):
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(62, 100)
        self.l2 = nn.Linear(100, 75)
        self.l3 = nn.Linear(75, 3)
    
    def forward(self, input_features):
        x = self.relu(self.l1(input_features))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        return x



class ChooseTrump(ChoiceNN):
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(57, 50)
        self.l2 = nn.Linear(50, 4)
    
    def forward(self, input_features):
        x = self.relu(self.l1(input_features))
        x = self.softmax(self.l2(x))
        return x

        

class ChooseCard(ChoiceNN):
    
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, int(input_dim*1.25))
        self.l2 = nn.Linear(int(input_dim*1.25), 75)
        self.l3 = nn.Linear(75, 57)
    
    def forward(self, input_features):
        x = self.relu(self.l1(input_features))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        return x
    


class Game:

    def __init__(self, load_params=False, verbose=False):
        """The basic game object that controls players, cards, & player decisions and plays games

        Parameters
        ----------
        load_params : bool, optional
            Whether to load starting parameters from a file instead of using random values. The default is False.
        verbose : bool, optional
            Whether to print game state updates while running. The default is False.

        """
        # set up neural nets
        self.chooseTrump = ChooseTrump()
        self.chooseWidow = ChooseCard(61)  # input dim is 4 + 57
        self.chooseBid = ChooseBid()
        # card choice nets for even team
        self.firstEvenChooseCard = ChooseCard(118)  # input dim is 4 + 2*57
        self.secondEvenChooseCard = ChooseCard(175)  # input dim is 4 + 3*57
        self.thirdEvenChooseCard = ChooseCard(175)
        self.fourthEvenChooseCard = ChooseCard(118)
        # card choice nets for odd team
        self.firstOddChooseCard = ChooseCard(118)
        self.secondOddChooseCard = ChooseCard(175)
        self.thirdOddChooseCard = ChooseCard(175)
        self.fourthOddChooseCard = ChooseCard(118)
        if load_params:
            self.load_params()
        # set up the game state to be ready for the start of the game
        self.reset()
        # create game_balance list to keep track of which team is winning the most games
        self.game_balance = []
        # set verbose param to determine whether to print game state updates
        self.verbose = verbose
        
    def load_params(self):
        with open(PARAMS, 'rb') as fh:
            params = pickle.load(fh)
        for obj, params_ in zip((self.chooseTrump, self.chooseWidow, self.chooseBid, self.firstEvenChooseCard,
                                 self.secondEvenChooseCard, self.thirdEvenChooseCard, self.fourthEvenChooseCard,
                                 self.firstOddChooseCard, self.secondOddChooseCard, self.thirdOddChooseCard,
                                 self.fourthOddChooseCard),
                                params):
            for p, param in zip(obj.parameters(), params_):
                p.data = param
        
    def dump_params(self):
        params = []
        for obj in (self.chooseTrump, self.chooseWidow, self.chooseBid, self.firstEvenChooseCard,
                    self.secondEvenChooseCard, self.thirdEvenChooseCard, self.fourthEvenChooseCard,
                    self.firstOddChooseCard, self.secondOddChooseCard, self.thirdOddChooseCard,
                    self.fourthOddChooseCard):
            params.append([p.data for p in obj.parameters()])
        with open(PARAMS, 'wb') as fh:
            pickle.dump(params, fh)
    
    def reset(self):
        # create list of all cards in the deck
        cards = [Card(color, number) for color in ['red', 'yellow', 'green', 'black'] for number in range(1, 15)] \
                + [Rook()]
        # shuffle deck and deal
        random.shuffle(cards)
        self.players = [cards[13*i:13*(i+1)] for i in range(4)]
        self.widow = cards[13*4:]
        # do bidding
        self.bid()
        # take widow and choose trump color
        self.choose_trump()
        self.trump_vec = color_to_one_hot(self.trump)
        self.score0 = 0  # score for players 0 and 2 (team 0)
        self.score1 = 0  # score for players 1 and 3 (team 1)
        self.cards_balance = 0  # difference of cards taken by team 0 and by team 1

    def bid(self):
        bid_amount = 0.0  # bid amounts scaled between 0 and 1 but represent numbers from 95 to 195
        # Features:
            # Current bid amount
            # Team member's last bid
            # Players still bidding
            # Cards in hand
        players_bidding = list(range(4))
        last_bids = {x: 0 for x in players_bidding}
        i = 0
        while True:
            current_bidder = players_bidding[i]
            # assemble features
            teammate_bid = last_bids[(i + 2) % 4]
            still_bidding = torch.zeros(3)
            for p in players_bidding:
                if p != current_bidder:
                    still_bidding[(p - current_bidder) % 4 - 1] = 1
            cards_in_hand = vectorize_cards(self.players[current_bidder])
            all_features = torch.cat((torch.tensor([bid_amount, teammate_bid]), still_bidding, cards_in_hand))
            action_vec = self.chooseBid(all_features)
            # update gradients
            action_vec.max().backward()
            self.chooseBid.cache_grads()
            # Interpret action
            # Possible actions are pass (0), bid up 5 (1), or bid up 10 (2)
            action = action_vec.argmax()
            if action == 0:
                players_bidding = [p for p in players_bidding if p != current_bidder]
                # test exit condition: all but one player pass
                if len(players_bidding) == 1:
                    self.bid_amount = int((bid_amount + 1) * 100)
                    self.bid_winner = players_bidding[0]
                    return
                i %= len(players_bidding)
            elif action == 1:
                bid_amount += 0.05
                last_bids[current_bidder] = bid_amount
                i = (i + 1) % len(players_bidding)
            elif action == 2:
                bid_amount += 0.1
                last_bids[current_bidder] = bid_amount
                i = (i + 1) % len(players_bidding)
            if bid_amount > 1:
                # test exit condition: bid reaches 200
                self.bid_amount = 200
                self.bid_winner = current_bidder
                return

    def choose_trump(self):
        """Given cards in hand and in widow, chooses trump color and exchanges cards with widow
        """
        # Ask the neural net for advice ;)
        all_cards = self.players[self.bid_winner] + self.widow
        all_cards_vec = vectorize_cards(all_cards)
        trump_color_vec = self.chooseTrump(all_cards_vec)
        # update gradients for chooseTrump neural net
        trump_color_vec.max().backward()
        self.chooseTrump.cache_grads()
        # Set self.trump
        max_idx = trump_color_vec.argmax()
        self.trump = 'red' if max_idx == 0 else 'yellow' if max_idx == 1 else 'green' if max_idx == 2 else 'black'
        # Now choose cards to put back in the widow
        widow_vec = self.chooseWidow(torch.cat((color_to_one_hot(self.trump), all_cards_vec)))
        allowed = all_cards_vec * widow_vec.detach()
        choice_idxs = torch.topk(allowed, 5).indices
        # update gradients...
        widow_vec[choice_idxs].mean().backward()
        self.chooseWidow.cache_grads()
        # Set widow and player's hand
        choice_dicts = [card_from_index(idx.item()).__dict__ for idx in choice_idxs]
        self.widow = [card for card in all_cards if card.__dict__ in choice_dicts]
        self.players[self.bid_winner] = [card for card in all_cards if card not in self.widow]

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
        """Choose a card to play.

        Parameters
        ----------
        cards : list of Card objects
            The cards that have already been played in the given hand (should have length from 0 to 3).
        player_order : list of ints
            The player order for the given round of play. Should be some ordered permutation of [0, 1, 2, 3].

        Returns
        -------
        Card object -- the card from the player's hand to be played.

        """
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
            card_choice_vec = self.firstEvenChooseCard(input_features) if player % 2 == 0 \
                                else self.firstOddChooseCard(input_features)
        elif num_cards_down == 1:
            possible_plays = vectorize_cards(sum(
                        (self.players[p] for p in player_order[2:]), []
                    ))
            if player != self.bid_winner:
                possible_plays += vectorize_cards(self.widow)
            cards_down = vectorize_cards(cards)
            input_features = torch.cat((self.trump_vec, cards_in_hand, possible_plays, cards_down))
            card_choice_vec = self.secondEvenChooseCard(input_features) if player % 2 == 0 \
                                else self.secondOddChooseCard(input_features)
        elif num_cards_down == 2:
            possible_plays = vectorize_cards(self.players[player_order[3]])
            if player != self.bid_winner:
                possible_plays += vectorize_cards(self.widow)
            cards_down = vectorize_cards(cards)
            input_features = torch.cat((self.trump_vec, cards_in_hand, possible_plays, cards_down))
            card_choice_vec = self.thirdEvenChooseCard(input_features) if player % 2 == 0 \
                                else self.thirdOddChooseCard(input_features)
        elif num_cards_down == 3:
            cards_down = vectorize_cards(cards)
            input_features = torch.cat((self.trump_vec, cards_in_hand, cards_down))
            card_choice_vec = self.fourthEvenChooseCard(input_features) if player % 2 == 0 \
                                else self.fourthOddChooseCard(input_features)
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
            if player % 2 == 0:
                self.firstEvenChooseCard.cache_grads()
            else:
                self.firstOddChooseCard.cache_grads()
        elif num_cards_down == 1:
            if player % 2 == 0:
                self.secondEvenChooseCard.cache_grads()
            else:
                self.secondOddChooseCard.cache_grads()
        elif num_cards_down == 2:
            if player % 2 == 0:
                self.thirdEvenChooseCard.cache_grads()
            else:
                self.thirdOddChooseCard.cache_grads()
        elif num_cards_down == 3:
            if player % 2 == 0:
                self.fourthEvenChooseCard.cache_grads()
            else:
                self.fourthOddChooseCard.cache_grads()
        # interpret the best card index as a Card and update the game state
        choice_dict = card_from_index(best_card_index).__dict__
        for i, card in enumerate(self.players[player]):
            if card.__dict__ == choice_dict:
                return self.players[player].pop(i)
        assert False, (available_plays, choice_dict, [c.__dict__ for c in self.players[player]])

    def play_hand(self, starting_player):
        """Goes through a hand of play (all four players play a single card)
        starting_player is an int from 0 to 3 inclusive denoting which player plays first in the hand
        """
        player_order = [(i+starting_player) % 4 for i in range(4)]
        cards = []
        for player in player_order:
            cards.append(self.choose_card(cards, player_order))
            if self.verbose:
                print(f'player {player} plays rook' if isinstance(cards[-1], Rook) else \
                      f'player {player} plays {cards[-1].color} {cards[-1].number}')
        winner, points = self.resolve_hand(cards, player_order)
        if winner % 2 == 0:
            self.score0 += points
        else:
            self.score1 += points
        return winner

    def play(self, train=False, decay=1):
        """Play a game (consisting of bidding and 13 rounds of play)

        Parameters
        ----------
        train : bool, optional
            Whether to update parameters based on the game's outcome. The default is False.
        decay : float [between 0 and 1], optional
            The learning rate decay to use for updating parameters. Only applicable if train is True.
            The default is 1 (i.e. no decay).

        """
        if self.verbose:
            print(f'Player {self.bid_winner} won the bid for {self.bid_amount}.')
            print(f'{self.trump} is trump')
        starting_player = self.bid_winner
        for i in range(13):
            if self.verbose:
                print(f'Round {i+1}:')
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
        if self.verbose:
            print(f'Score is {self.score0} to {self.score1}.')
            print('Even team wins.' if self.score0 > self.score1 else \
                  'Odd team wins.' if self.score0 < self.score1 else \
                  'It\'s a tie.')
        if train:
            lr = max(MAX_LR * decay, MIN_LR)
            # update neural net parameters based on acccumulated gradients & who won the game
            # if a player won, reinforce the params based on the grad, otherwise do opposite
            sign = 1 if self.score0 > self.score1 else -1 if self.score0 < self.score1 else 0
            self.firstEvenChooseCard.adam_update(sign * lr)
            self.secondEvenChooseCard.adam_update(sign * lr)
            self.thirdEvenChooseCard.adam_update(sign * lr)
            self.fourthEvenChooseCard.adam_update(sign * lr)
            self.firstOddChooseCard.adam_update(-1 * sign * lr)
            self.secondOddChooseCard.adam_update(-1 * sign * lr)
            self.thirdOddChooseCard.adam_update(-1 * sign * lr)
            self.fourthOddChooseCard.adam_update(-1 * sign * lr)
            # update params from pre-game phase
            sign *= (1 if self.bid_winner % 2 == 0 else -1)
            self.chooseBid.adam_update(sign * lr)
            self.chooseTrump.adam_update(sign * lr)
            self.chooseWidow.adam_update(sign * lr)
        # record game outcome
        self.game_balance.append(int(self.score0 > self.score1) if self.score0 != self.score1 else 0.5)
        # reset the game state to play again
        self.reset()
    
    def train(self, rounds=1000, dump=True):
        """Train the associated neural networks by playing games against themselves.

        Parameters
        ----------
        rounds : [positive] int, optional
            The number of training rounds to play. It takes about 5 sec for each 100 rounds. The default is 10000.
        dump : bool, optional
            Whether to save the post-training neural net parameters to a file that can be loaded later.
            The default is True.

        """
        time0 = time()
        for i in range(rounds):
            self.play(train=True, decay=(exp(-i*DECAY_RATE)))
            if (i+1) % 100 == 0 or (i+1) == rounds:
                print(f'Completed {i+1} / {rounds} training rounds (average {(time()-time0)*10:.2f} ms per round).')
                time0 = time()
        if dump:
            self.dump_params()



game = Game()
game.train()
