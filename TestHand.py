from DeepPeg import DeepPeg
from Deck import Card, Rank, Suit

if __name__ == '__main__':
    player = DeepPeg(1, True, False, True)
    player.playhand = [Card(Rank.Ace, Suit.Hearts), Card(Rank.Two, Suit.Spades)]
    state = dict()
    scores = [0, 0]
    state['scores'] = scores
    numCards = [2, 1]
    state['numCards'] = numCards
    inplay = [Card(Rank.Ten, Suit.Clubs), Card(Rank.Three, Suit.Hearts)]
    state['inplay'] = inplay
    state['playorder'] = inplay
    state['dealer'] = 0
    state['starter'] = Card(Rank.Ace, Suit.Spades)
    state['count'] = sum([card.value() for card in inplay])

    played = player.playCard(state, Card(Rank.Ace, Suit.Hearts))
    print(played)
