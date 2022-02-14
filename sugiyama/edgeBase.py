class EdgeBase(object):
    """Base class for edges.

       Attributes:
          v (list[Vertex]): list of vertices associated with this edge.
          degree (int): degree of the edge (number of unique vertices).
    """

    def __init__(self, x, y):
        self.degree = 0 if x == y else 1
        self.v = (x, y)