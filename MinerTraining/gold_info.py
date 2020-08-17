class GoldInfo:
    def __init__(self):
        self.posx = 0
        self.posy = 0
        self.amount = 0

    def loads(self, data):
        golds = []
        for gd in data:
            g = GoldInfo()
            g.posx = gd["posx"]
            g.posy = gd["posy"]
            g.amount = gd["amount"]
            golds.append(g)
        return golds
