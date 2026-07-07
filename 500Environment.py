class Card:
    rank: int
    suit: int
    joker: bool

    def __init__(self, rank, suit, joker=False) -> None:
        self.rank = rank
        self.suit = suit
        self.joker = joker
    
    def id(self) -> int:
        if self.joker:
            return 43
        else:
            return -1