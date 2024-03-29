from sugiyama.vertexBase import VertexBase


class Vertex(VertexBase):
    """Vertex class enhancing a VertexBase with graph-related features.

       Attributes:
          c (GraphBase): the component of connected vertices that contains this vertex.
             By default a vertex belongs no component but when it is added in a
             graph, c points to the connected component in this graph.
          data (object) : an object associated with the vertex.
    """

    def __init__(self, data=None):
        super().__init__()
        # by default, a new vertex belongs to its own component
        # but when the vertex is added to a graph, c points to the
        # connected component where it belongs.
        self.c = None
        self.data = data
        self.__index = None

    @property
    def index(self):
        from sugiyama.graphBase import GraphBase
        if self.__index:
            return self.__index
        elif isinstance(self.c, GraphBase):
            self.__index = self.c.sV.index(self)
            return self.__index
        else:
            return None

    def __lt__(self, v):
        return 0

    def __gt__(self, v):
        return 0

    def __le__(self, v):
        return 0

    def __ge__(self, v):
        return 0

    def __getstate__(self):
        return (self.index, self.data)

    def __setstate__(self, state):
        self.__index, self.data = state
        self.c = None
        self.e = []