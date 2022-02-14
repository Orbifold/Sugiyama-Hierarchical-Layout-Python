class EdgeBase(object):
    """The Edge essentials attributes.

       Attributes:
          v (list[Vertex]): list of vertices associated with this edge.
          deg (int): degree of the edge (number of unique vertices).
    """

    def __init__(self, x, y):
        self.deg = 0 if x == y else 1
        self.v = (x, y)