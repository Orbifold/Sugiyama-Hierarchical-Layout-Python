import pytest

import sugiyama.edge
import sugiyama.vertex
import sugiyama.vertexViewer
from sugiyama import *
from sugiyama.sugiyamaLayout import SugiyamaLayout
@pytest.mark.skipif(not utils.dot._has_ply,reason="requires ply module to parse dot input file")
def test_cycles(sample_cycle):
    g = utils.Dot().read(sample_cycle)[0]
    V = {}
    for k,v in g.nodes.items():
        V[k]= sugiyama.vertex.Vertex(k)
        V[k].view = sugiyama.vertexViewer.VertexViewer(10, 10)
    E = []
    for e in g.edges:
        E.append(sugiyama.edge.Edge(V[e.n1.name], V[e.n2.name]))

    G = graph.Graph(V.values(), E)
    assert len(G.C)==1
    sg = SugiyamaLayout(G.C[0])
    gr = sg.g

    r = gr.roots()
    assert len(r)==2
    assert V['A1'] in r
    assert V['A2'] in r

    L = gr.get_scs_with_feedback(r)
    assert len(L)==5
    for s in L:
        if V['A1'] in s:
            assert len(s)==1
        if V['A2'] in s:
            assert len(s)==1
        if V['B1'] in s:
            assert len(s)==1
        if V['B2'] in s:
            assert len(s)==1
        if len(s)>1:
            assert V['C1'] in s
            assert V['C2'] in s
            assert V['D1'] in s
            assert V['D2'] in s
            assert len(s)==4

def test_longcycle():
    V = {}
    for x in 'abcdefgh':
        v = sugiyama.vertex.Vertex(x)
        v.view = sugiyama.vertexViewer.VertexViewer()
        V[x] = v
    E = []
    for e in ('ab','bc','cd','de','eb','bf','dg','gh','fh'):
        E.append(sugiyama.edge.Edge(V[e[0]], V[e[1]]))
    G = graph.Graph(V.values(), E)
    l = SugiyamaLayout(G.C[0])
    l.init_all()
    assert len(l.alt_e)==1
    assert l.alt_e[0] == E[4]
    assert l.grx[V['e']].rank == 4
    assert sum((v.dummy for v in l.grx.values()))==4
