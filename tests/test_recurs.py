import pytest

from sugiyama.edge import Edge
from  sugiyama.graph  import *
from  sugiyama.layouts import *
from sugiyama.vertex import Vertex


@pytest.mark.xfail
@pytest.mark.skip(reason="to reconsider")
def test_recurs():
    # Note, this is failing for me (fabioz) with: RuntimeError: maximum recursion depth exceeded in cmp
    # => adjusting recursion depth dynamically works only with CPython
    # TODO: reimplement the recursive parts of SugiyamaLayout in iterative form.
    v = range(1001)
    V = [Vertex(x) for x in v]
    E = [Edge(V[x],V[x+1]) for x in range(1000)]

    gr  = GraphBase(V, E)
    for  v in gr.V(): v.view = VertexViewer(10,10)
    sug  = SugiyamaLayout(gr)
    sug.init_all()
    sug.draw(1)
