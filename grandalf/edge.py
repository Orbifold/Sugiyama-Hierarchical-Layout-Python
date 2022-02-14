from grandalf.edgeBase import EdgeBase


class Edge(EdgeBase):
    """Edge class enhancing EdgeBase with attributes and methods related to the graph.

       Attributes:
         w (int): a weight associated with the edge (default 1) used by Dijkstra to
           find min-flow paths.
         data (object): an object associated with the edge.
         feedback (bool): indicates if the edge has been marked as a *feeback* edge
           by the Tarjan algorithm which means that it is part of a cycle and that
           inverting this edge would remove this cycle.

       Methods:
         attach(): add this edge in its vertices edge lists.
         detach(): remove this edge from its vertices edge lists.
    """

    def __init__(self, x, y, w=1, data=None, connect=False):
        super().__init__(x, y)
        # w is an optional weight associated with the edge.
        self.w = w
        self.data = data
        self.feedback = False
        if connect and (x.c is None or y.c is None):
            c = x.c or y.c
            c.add_edge(self)

    def attach(self):
        if not self in self.v[0].e:
            self.v[0].e.append(self)
        if not self in self.v[1].e:
            self.v[1].e.append(self)

    def detach(self):
        if self.deg == 1:
            assert self in self.v[0].e
            assert self in self.v[1].e
            self.v[0].e.remove(self)
            self.v[1].e.remove(self)
        else:
            if self in self.v[0].e:
                self.v[0].e.remove(self)
            assert self not in self.v[0].e
        return [self]

    def __lt__(self, v):
        return 0

    def __gt__(self, v):
        return 0

    def __le__(self, v):
        return 0

    def __ge__(self, v):
        return 0

    def __getstate__(self):
        xi, yi = (self.v[0].index, self.v[1].index)
        return (xi, yi, self.w, self.data, self.feedback)

    def __setstate__(self, state):
        xi, yi, self.w, self.data, self.feedback = state
        self._v = [xi, yi]
        self.deg = 0 if xi == yi else 1