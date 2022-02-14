class LayoutVertex(object):
    """
    The sugiyama layout adds new attributes to vertices.
    These attributes are stored in an internal _sugimyama_vertex_attr object.

    Attributes:
        rank (int): the rank number is the index of the layer that
                    contains this vertex.
        dummy (0/1): a flag indicating if the vertex is *dummy*
        pos (int): the index of the vertex in the layer
        x (list(float)): the list of computed horizontal coordinates of the vertex
        bar (float): the current *barycenter* of the vertex
    """

    def __init__(self, r=None, d=0):
        self.rank = r
        self.dummy = d
        self.pos = None
        self.x = 0
        self.bar = None

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.rank, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s

    # def __eq__(self,x):
    #    return self.bar == x.bar
    # def __ne__(self,x):
    #    return self.bar != x.bar
    # def __lt__(self,x):
    #    return self.bar < x.bar
    # def __le__(self,x):
    #    return self.bar <= x.bar
    # def __gt__(self,x):
    #    return self.bar > x.bar
    # def __ge__(self,x):
    #    return self.bar >= x.bar