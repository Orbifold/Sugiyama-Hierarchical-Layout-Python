# -*- coding: utf-8 -*-


from sys import getrecursionlimit, setrecursionlimit

from sugiyama.dummyVertex import DummyVertex
from sugiyama.layer import Layer
from sugiyama.layoutVertex import LayoutVertex


class SugiyamaLayout(object):
    """
    The Sugiyama layout is the traditional "layered" graph layout called
    *dot* in graphviz. This layout is quite efficient but heavily relies 
    on drawing heuristics. Adaptive drawing is limited to
    extending the leaves only, but since the algorithm is quite fast
    redrawing the entire graph (up to about a thousand nodes) gives
    usually good results in less than a second.

    The Sugiyama Layout Class takes as input a core_graph object and implements
    an efficient drawing algorithm based on nodes dimensions provided through
    a user-defined *view* property in each vertex.

    Attributes:
        dirvh (int): the current aligment state
                     for alignment policy:
                     dirvh=0 -> dirh=+1, dirv=-1: leftmost upper
                     dirvh=1 -> dirh=-1, dirv=-1: rightmost upper
                     dirvh=2 -> dirh=+1, dirv=+1: leftmost lower
                     dirvh=3 -> dirh=-1, dirv=+1: rightmost lower
        order_inter (int): the default number of layer placement iterations
        order_attr (str): set attribute name used for layer ordering
        xspace (int): horizontal space between vertices in a layer
        yspace (int): vertical space between layers
        dw (int): default width of a vertex
        dh (int): default height of a vertex
        g (GraphBase): the graph component reference
        layers (list[sugiyama.layer.Layer]): the list of layers
        grx (dict): associate vertex (possibly dummy) with their sugiyama attributes
        ctrls (dict): associate edge with all its vertices (including dummies)
        dag (bool): the current acyclic state 
        initdone (bool): True if state is initialized (see init_all).
    """

    def __init__(self, g):
        from sugiyama.utils.geometry import median_wh

        # drawing parameters:
        self.dirvh = 0
        self.order_iter = 8
        self.order_attr = "pos"
        self.xspace = 20
        self.yspace = 20
        self.dw = 10
        self.dh = 10
        # For layered graphs, vertices and edges need to have some additional
        # attributes that make sense only for this kind of layout:
        # update graph struct:
        self.g = g
        self.layers = []
        self.grx = {}
        self.ctrls = {}
        self.dag = False
        for v in self.g.V():
            assert hasattr(v, "view")
            self.grx[v] = LayoutVertex()
        self.dw, self.dh = median_wh([v.view for v in self.g.V()])
        self.initdone = False

    def init_all(self, roots=None, inverted_edges=None, optimize=False):
        """initializes the layout algorithm by computing roots (unless provided),
           inverted edges (unless provided), vertices ranks and creates all dummy
           vertices and layers. 
             
             Parameters:
                roots (list[Vertex]): set *root* vertices (layer 0)
                inverted_edges (list[Edge]): set edges to invert to have a DAG.
                optimize (bool): optimize ranking if True (default False)
        """
        if self.initdone:
            return
        # For layered sugiyama algorithm, the input graph must be acyclic,
        # so we must provide a list of root nodes and a list of inverted edges.
        if roots is None:
            roots = [v for v in self.g.sV if len(v.e_in()) == 0]
        if inverted_edges is None:
            _ = self.g.get_scs_with_feedback(roots)
            inverted_edges = [x for x in self.g.sE if x.feedback]
        self.alt_e = inverted_edges
        # assign rank to all vertices:
        self.rank_all(roots, optimize)
        # add dummy vertex/edge for 'long' edges:
        for e in self.g.E():
            self.setdummies(e)
        # precompute some layers values:
        for l in self.layers:
            l.setup(self)
        self.initdone = True

    def draw(self, N=1.5):
        """compute every node coordinates after converging to optimal ordering by N
           rounds, and finally perform the edge routing.
        """
        while N > 0.5:
            for (l, mvmt) in self.ordering_step():
                pass
            N = N - 1
        if N > 0:
            for (l, mvmt) in self.ordering_step(oneway=True):
                pass
        self.setxy()
        self.draw_edges()

    def _edge_inverter(self):
        for e in self.alt_e:
            x, y = e.v
            e.v = (y, x)
        self.dag = not self.dag
        if self.dag:
            for e in self.g.degenerated_edges:
                e.detach()
                self.g.sE.remove(e)
        else:
            for e in self.g.degenerated_edges:
                self.g.add_edge(e)

    @property
    def dirvh(self):
        return self.__dirvh

    @property
    def dirv(self):
        return self.__dirv

    @property
    def dirh(self):
        return self.__dirh

    @dirvh.setter
    def dirvh(self, dirvh):
        assert dirvh in range(4)
        self.__dirvh = dirvh
        self.__dirh, self.__dirv = {0: (1, -1),
                                    1: (-1, -1),
                                    2: (1, 1),
                                    3: (-1, 1)}[dirvh]

    @dirv.setter
    def dirv(self, dirv):
        assert dirv in (-1, +1)
        dirvh = (dirv + 1) + (1 - self.__dirh) // 2
        self.dirvh = dirvh

    @dirh.setter
    def dirh(self, dirh):
        assert dirh in (-1, +1)
        dirvh = (self.__dirv + 1) + (1 - dirh) // 2
        self.dirvh = dirvh

    def rank_all(self, roots, optimize=False):
        """Computes rank of all vertices.
        add provided roots to rank 0 vertices,
        otherwise update ranking from provided roots.
        The initial rank is based on precedence relationships,
        optimal ranking may be derived from network flow (simplex).
        """
        self._edge_inverter()
        r = [x for x in self.g.sV if (len(x.e_in()) == 0 and x not in roots)]
        self._rank_init(roots + r)
        if optimize:
            self._rank_optimize()
        self._edge_inverter()

    def _rank_init(self, unranked):
        """Computes rank of provided unranked list of vertices and all
           their children. A vertex will be asign a rank when all its 
           inward edges have been *scanned*. When a vertex is asigned
           a rank, its outward edges are marked *scanned*.
        """
        assert self.dag
        scan = {}
        # set rank of unranked based on its in-edges vertices ranks:
        while len(unranked) > 0:
            l = []
            for v in unranked:
                self.setrank(v)
                # mark out-edges has scan-able:
                for e in v.e_out():
                    scan[e] = True
                # check if out-vertices are rank-able:
                for x in v.neighbors(+1):
                    if not (False in [scan.get(e, False) for e in x.e_in()]):
                        if x not in l:
                            l.append(x)
            unranked = l

    def _rank_optimize(self):
        """optimize ranking by pushing long edges toward lower layers as much as possible.
        see other interersting network flow solver to minimize total edge length
        (http://jgaa.info/accepted/2005/EiglspergerSiebenhallerKaufmann2005.9.3.pdf)
        """
        assert self.dag
        for l in reversed(self.layers):
            for v in l:
                gv = self.grx[v]
                for x in v.neighbors(-1):
                    if all((self.grx[y].rank >= gv.rank for y in x.neighbors(+1))):
                        gx = self.grx[x]
                        self.layers[gx.rank].remove(x)
                        gx.rank = gv.rank - 1
                        self.layers[gv.rank - 1].append(x)

    def setrank(self, v):
        """set rank value for vertex v and add it to the corresponding layer.
           The Layer is created if it is the first vertex with this rank.
        """
        assert self.dag
        r = max([self.grx[x].rank for x in v.neighbors(-1)] + [-1]) + 1
        self.grx[v].rank = r
        # add it to its layer:
        try:
            self.layers[r].append(v)
        except IndexError:
            assert r == len(self.layers)
            self.layers.append(Layer([v]))

    def dummyctrl(self, r, ctrl):
        """creates a DummyVertex at rank r inserted in the ctrl dict
           of the associated edge and layer.

           Arguments:
              r (int): rank value
              ctrl (dict): the edge's control vertices
           
           Returns:
              sugiyama.dummyVertex.DummyVertex : the created DummyVertex.
        """
        dv = DummyVertex(r)
        dv.view.w, dv.view.h = self.dw, self.dh
        self.grx[dv] = dv
        dv.ctrl = ctrl
        ctrl[r] = dv
        self.layers[r].append(dv)
        return dv

    def setdummies(self, e):
        """creates and defines all needed dummy vertices for edge e.
        """
        v0, v1 = e.v
        r0, r1 = self.grx[v0].rank, self.grx[v1].rank
        if r0 > r1:
            assert e in self.alt_e
            v0, v1 = v1, v0
            r0, r1 = r1, r0
        if (r1 - r0) > 1:
            # "dummy vertices" are stored in the edge ctrl dict,
            # keyed by their rank in layers.
            ctrl = self.ctrls[e] = {}
            ctrl[r0] = v0
            ctrl[r1] = v1
            for r in range(r0 + 1, r1):
                self.dummyctrl(r, ctrl)

    def draw_step(self):
        """iterator that computes all vertices coordinates and edge routing after
           just one step (one layer after the other from top to bottom to top).
           Purely inefficient ! Use it only for "animation" or debugging purpose.
        """
        ostep = self.ordering_step()
        for s in ostep:
            self.setxy()
            self.draw_edges()
            yield s

    def ordering_step(self, oneway=False):
        """iterator that computes all vertices ordering in their layers
           (one layer after the other from top to bottom, to top again unless
           oneway is True).
        """
        self.dirv = -1
        crossings = 0
        for l in self.layers:
            mvmt = l.order()
            crossings += mvmt
            yield (l, mvmt)
        if oneway or (crossings == 0):
            return
        self.dirv = +1
        while l:
            mvmt = l.order()
            yield (l, mvmt)
            l = l.nextlayer()

    def setxy(self):
        """computes all vertex coordinates (x,y) using
        an algorithm by Brandes & Kopf.
        """
        self._edge_inverter()
        self._detect_alignment_conflicts()
        inf = float("infinity")
        # initialize vertex coordinates attributes:
        for l in self.layers:
            for v in l:
                self.grx[v].root = v
                self.grx[v].align = v
                self.grx[v].sink = v
                self.grx[v].shift = inf
                self.grx[v].X = None
                self.grx[v].x = [0.0] * 4
        curvh = self.dirvh  # save current dirvh value
        for dirvh in range(4):
            self.dirvh = dirvh
            self._coord_vertical_alignment()
            self._coord_horizontal_compact()
        self.dirvh = curvh  # restore it
        # vertical coordinate assigment of all nodes:
        Y = 0
        for l in self.layers:
            dY = max([v.view.h / 2.0 for v in l])
            for v in l:
                vx = sorted(self.grx[v].x)
                # mean of the 2 medians out of the 4 x-coord computed above:
                avgm = (vx[1] + vx[2]) / 2.0
                # final xy-coordinates :
                v.view.xy = (avgm, Y + dY)
            Y += 2 * dY + self.yspace
        self._edge_inverter()

    def _detect_alignment_conflicts(self):
        """mark conflicts between edges:
        inner edges are edges between dummy nodes
        type 0 is regular crossing regular (or sharing vertex)
        type 1 is inner crossing regular (targeted crossings)
        type 2 is inner crossing inner (avoided by reduce_crossings phase)
        """
        curvh = self.dirvh  # save current dirvh value
        self.dirvh = 0
        self.conflicts = []
        for L in self.layers:
            last = len(L) - 1
            prev = L.prevlayer()
            if not prev:
                continue
            k0 = 0
            k1_init = len(prev) - 1
            l = 0
            for l1, v in enumerate(L):
                if not self.grx[v].dummy:
                    continue
                if l1 == last or v.inner(-1):
                    k1 = k1_init
                    if v.inner(-1):
                        k1 = self.grx[v.neighbors(-1)[-1]].pos
                    for vl in L[l : l1 + 1]:
                        for vk in L._neighbors(vl):
                            k = self.grx[vk].pos
                            if k < k0 or k > k1:
                                self.conflicts.append((vk, vl))
                    l = l1 + 1
                    k0 = k1
        self.dirvh = curvh  # restore it

    def _coord_vertical_alignment(self):
        """performs vertical alignment according to current dirvh internal state.
        """
        dirh, dirv = self.dirh, self.dirv
        g = self.grx
        for l in self.layers[::-dirv]:
            if not l.prevlayer():
                continue
            r = None
            for vk in l[::dirh]:
                for m in l._medianindex(vk):
                    # take the median node in dirv layer:
                    um = l.prevlayer()[m]
                    # if vk is "free" align it with um's root
                    if g[vk].align is vk:
                        if dirv == 1:
                            vpair = (vk, um)
                        else:
                            vpair = (um, vk)
                        # if vk<->um link is used for alignment
                        if (vpair not in self.conflicts) and (
                            (r is None) or (dirh * r < dirh * m)
                        ):
                            g[um].align = vk
                            g[vk].root = g[um].root
                            g[vk].align = g[vk].root
                            r = m

    def _coord_horizontal_compact(self):
        limit = getrecursionlimit()
        N = len(self.layers) + 10
        if N > limit:
            setrecursionlimit(N)
        dirh, dirv = self.dirh, self.dirv
        g = self.grx
        L = self.layers[::-dirv]
        # recursive placement of blocks:
        for l in L:
            for v in l[::dirh]:
                if g[v].root is v:
                    self.__place_block(v)
        setrecursionlimit(limit)
        # mirror all nodes if right-aligned:
        if dirh == -1:
            for l in L:
                for v in l:
                    x = g[v].X
                    if x:
                        g[v].X = -x
        # then assign x-coord of its root:
        inf = float("infinity")
        rb = inf
        for l in L:
            for v in l[::dirh]:
                g[v].x[self.dirvh] = g[g[v].root].X
                rs = g[g[v].root].sink
                s = g[rs].shift
                if s < inf:
                    g[v].x[self.dirvh] += dirh * s
                rb = min(rb, g[v].x[self.dirvh])
        # normalize to 0, and reinit root/align/sink/shift/X
        for l in self.layers:
            for v in l:
                # g[v].x[dirvh] -= rb
                g[v].root = g[v].align = g[v].sink = v
                g[v].shift = inf
                g[v].X = None

    # TODO: rewrite in iterative form to avoid recursion limit...
    def __place_block(self, v):
        g = self.grx
        if g[v].X is None:
            # every block is initially placed at x=0
            g[v].X = 0.0
            # place block in which v belongs:
            w = v
            while 1:
                j = g[w].pos - self.dirh  # predecessor in rank must be placed
                r = g[w].rank
                if 0 <= j < len(self.layers[r]):
                    wprec = self.layers[r][j]
                    delta = (
                        self.xspace + (wprec.view.w + w.view.w) / 2.0
                    )  # abs positive minimum displ.
                    # take root and place block:
                    u = g[wprec].root
                    self.__place_block(u)
                    # set sink as sink of prec-block root
                    if g[v].sink is v:
                        g[v].sink = g[u].sink
                    if g[v].sink != g[u].sink:
                        s = g[u].sink
                        newshift = g[v].X - (g[u].X + delta)
                        g[s].shift = min(g[s].shift, newshift)
                    else:
                        g[v].X = max(g[v].X, (g[u].X + delta))
                # take next node to align in block:
                w = g[w].align
                # quit if self aligned
                if w is v:
                    break

    def draw_edges(self):
        """Basic edge routing applied only for edges with dummy points.
        Enhanced edge routing can be performed by using the apropriate
        *route_with_xxx* functions from :ref:routing_ in the edges' view.
        """
        for e in self.g.E():
            if hasattr(e, "view"):
                l = []
                if e in self.ctrls:
                    D = self.ctrls[e]
                    r0, r1 = self.grx[e.v[0]].rank, self.grx[e.v[1]].rank
                    if r0 < r1:
                        ranks = range(r0 + 1, r1)
                    else:
                        ranks = range(r0 - 1, r1, -1)
                    l = [D[r].view.xy for r in ranks]
                l.insert(0, e.v[0].view.xy)
                l.append(e.v[1].view.xy)
                try:
                    self.route_edge(e, l)
                except AttributeError:
                    pass
                e.view.setpath(l)

