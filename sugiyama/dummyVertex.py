from sugiyama.layoutVertex import LayoutVertex
from sugiyama.vertexViewer import VertexViewer


class DummyVertex(LayoutVertex):
    """
    The DummyVertex class is used by the sugiyama layout to represent
    *long* edges, i.e. edges that span over several ranks.
    For these edges, a DummyVertex is inserted in every inner layer.

    Attributes:
        view (viewclass): since a DummyVertex is acting as a Vertex, it
                          must have a view.
        ctrl (list[_sugiyama_attr]): the list of associated dummy vertices

    Methods:
        neighbors(dir): reflect the Vertex method and returns the list of adjacent
                 vertices (possibly dummy) in the given direction.
        inner(dir): return True if a neighbor in the given direction is *dummy*.
    """

    def __init__(self, r = None, viewclass = VertexViewer):
        self.view = viewclass()
        self.ctrl = None
        super().__init__(r, d = 1)

    def neighbors(self, dir):
        assert dir == +1 or dir == -1
        v = self.ctrl.get(self.rank + dir, None)
        return [v] if v is not None else []

    def inner(self, dir):
        assert dir == +1 or dir == -1
        try:
            return any([x.dummy == 1 for x in self.neighbors(dir)])
        except KeyError:
            return False
        except AttributeError:
            return False

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.rank, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s
