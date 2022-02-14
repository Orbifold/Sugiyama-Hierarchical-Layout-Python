from sugiyama.utils import Poset
from sugiyama.vertex import Vertex


class GraphBase(object):
    """A connected graph of Vertex/Edge objects. A GraphBase is a *component*
       of a Graph that contains a connected set of Vertex and Edges.

       Attributes:
         sV (poset[Vertex]): the partially ordered set of vertices of the graph.
         sE (poset[Edge]): the partially ordered set of edges of the graph.
         degenerated_edges (set[grandalf.edge.Edge]): the set of *degenerated* edges (of degree 0).
         directed (bool): indicates if the graph is considered *oriented* or not.

       Methods:
         V(cond=None): generates an iterator over vertices, with optional filter
         E(cond=None): generates an iterator over edges, with optional filter
         M(cond=None): returns the associativity matrix of the graph component
         order(): the order of the graph (number of vertices)
         norm(): the norm of the graph (number of edges)
         deg_min(): the minimum degree of vertices
         deg_max(): the maximum degree of vertices
         deg_avg(): the average degree of vertices
         eps(): the graph epsilon value (norm/order), average number of edges per vertex.
         path(x,y,f_io=0,hook=None): shortest path between vertices x and y by breadth-first descent,
           contrained by f_io direction if provided. The path is returned as a list of Vertex objects.
           If a *hook* function is provided, it is called at every vertex added to the path, passing
           the vertex object as argument.
         roots(): returns the list of *roots* (vertices with no inward edges).
         leaves(): returns the list of *leaves* (vertices with no outward edges).
         add_single_vertex(v): allow a GraphBase to hold a single vertex.
         add_edge(e): add edge e. At least one of its vertex must belong to the graph,
           the other being added automatically.
         remove_edge(e): remove Edge e, asserting that the resulting graph is still connex.
         remove_vertex(x): remove Vertex x and all associated edges.
         dijkstra(x,f_io=0,hook=None): shortest weighted-edges paths between x and all other vertices
           by dijkstra's algorithm with heap used as priority queue.
         get_scs_with_feedback(): returns the set of strongly connected components
           ("scs") by using Tarjan algorithm.
           These are maximal sets of vertices such that there is a path from each
           vertex to every other vertex.
           The algorithm performs a DFS from the provided list of root vertices.
           A cycle is of course a strongly connected component,
           but a strongly connected component can include several cycles.
           The Feedback Acyclic Set of edge to be removed/reversed is provided by
           marking the edges with a "feedback" flag.
           Complexity is O(V+E).
         partition(): returns a *partition* of the connected graph as a list of lists.
         N(v): returns neighbours of a vertex v.
    """

    def __init__(self, V=None, E=None, directed=True):
        if V is None:
            V = []
        if E is None:
            E = []
        self.directed = directed
        self.sV = Poset(V)
        self.sE = Poset([])

        self.degenerated_edges = set()

        if len(self.sV) == 1:
            v = self.sV[0]
            v.c = self
            for e in v.e:
                e.detach()
            return

        for e in E:
            x = self.sV.get(e.v[0])
            y = self.sV.get(e.v[1])
            if x is None or y is None:
                raise ValueError("unknown Vertex (%s or %s)" % e.v)
            e.v = (x, y)
            if e.deg == 0:
                self.degenerated_edges.add(e)
            e = self.sE.add(e)
            e.attach()
            if x.c is None:
                x.c = Poset([x])
            if y.c is None:
                y.c = Poset([y])
            if id(x.c) != id(y.c):
                x, y = (x, y) if len(x.c) > len(y.c) else (y, x)
                x.c.update(y.c)
                for v in y.c:
                    v.c = x.c
            s = x.c
        # check if graph is connected:
        for v in self.V():
            if v.c is None or (v.c != s):
                raise ValueError("unconnected Vertex %s" % v.data)
            else:
                v.c = self

    def roots(self):
        return list(filter(lambda v: len(v.e_in()) == 0, self.sV))

    def leaves(self):
        return list(filter(lambda v: len(v.e_out()) == 0, self.sV))

    def add_single_vertex(self, v):
        if len(self.sE) == 0 and len(self.sV) == 0:
            v = self.sV.add(v)
            v.c = self
            return v
        return None

    def add_edge(self, e):
        if e in self.sE:
            return self.sE.get(e)
        x = e.v[0]
        y = e.v[1]
        if not ((x in self.sV) or (y in self.sV)):
            raise ValueError("unconnected edge")
        x = self.sV.add(x)
        y = self.sV.add(y)
        e.v = (x, y)
        e.attach()
        e = self.sE.add(e)
        x.c = self
        y.c = self
        if e.deg == 0:
            self.degenerated_edges.add(e)
        return e

    def remove_edge(self, e):
        if not e in self.sE:
            return
        e.detach()
        # check if still connected (path is not oriented here):
        if e.deg == 1 and not self.path(e.v[0], e.v[1]):
            # return to inital state by reconnecting everything:
            e.attach()
            # exit with exception!
            raise ValueError(e)
        else:
            e = self.sE.remove(e)
            if e in self.degenerated_edges:
                self.degenerated_edges.remove(e)
            return e

    def remove_vertex(self, x):
        if x not in self.sV:
            return
        V = x.N()  # get all neighbor vertices to check paths
        E = x.detach()  # remove the edges from x and neighbors list
        # now we need to check if all neighbors are still connected,
        # and it is sufficient to check if one of them is connected to
        # all others:
        v0 = V.pop(0)
        for v in V:
            if not self.path(v0, v):
                # repair everything and raise exception if not connected:
                for e in E:
                    e.attach()
                raise ValueError(x)
        # remove edges and vertex from internal sets:
        for e in E:
            self.sE.remove(e)
        x = self.sV.remove(x)
        x.c = None
        return x

    def V(self, cond=None):
        V = self.sV
        if cond is None:
            cond = lambda x: True
        for v in V:
            if cond(v):
                yield v

    def E(self, cond=None):
        E = self.sE
        if cond is None:
            cond = lambda x: True
        for e in E:
            if cond(e):
                yield e

    def M(self, cond=None):
        from array import array

        mat = []
        for v in self.V(cond):
            vec = array("b", [0] * self.order())
            mat.append(vec)
            for e in v.e_in():
                v0 = e.v[0]
                if v0.index == v.index:
                    continue
                vec[v0.index] = -e.w
            for e in v.e_out():
                v1 = e.v[1]
                vec[v1.index] = e.w
        return mat

    def order(self):
        return len(self.sV)

    def norm(self):
        return len(self.sE)

    def deg_min(self):
        return min([v.deg() for v in self.sV])

    def deg_max(self):
        return max([v.deg() for v in self.sV])

    def deg_avg(self):
        return sum([v.deg() for v in self.sV]) / float(self.order())

    def eps(self):
        return float(self.norm()) / self.order()

    def path(self, x, y, f_io=0, hook=None):
        assert x in self.sV
        assert y in self.sV
        x = self.sV.get(x)
        y = self.sV.get(y)
        if x == y:
            return []
        if f_io != 0:
            assert self.directed == True
        # path:
        p = None
        if hook is None:
            hook = lambda x: False
        # apply hook:
        hook(x)
        # visisted:
        v = {x: None}
        # queue:
        q = [x]
        while (not p) and len(q) > 0:
            c = q.pop(0)
            for n in c.N(f_io):
                if not n in v:
                    hook(n)
                    v[n] = c
                    if n == y:
                        p = [n]
                    q.append(n)
                if p:
                    break
        # now we fill the path p backward from y to x:
        while p and p[0] != x:
            p.insert(0, v[p[0]])
        return p

    def dijkstra(self, x, f_io=0, hook=None):
        from collections import defaultdict
        from heapq import heappop, heappush

        if x not in self.sV:
            return None
        if f_io != 0:
            assert self.directed == True
        # initiate with path to itself...
        v = self.sV.get(x)
        # D is the returned vector of distances:
        D = defaultdict(lambda: None)
        D[v] = 0.0
        L = [(D[v], v)]
        while len(L) > 0:
            l, u = heappop(L)
            for e in u.e_dir(f_io):
                v = e.v[0] if (u is e.v[1]) else e.v[1]
                Dv = l + e.w
                if D[v] != None:
                    # check if heap/D needs updating:
                    # ignore if a shorter path was found already...
                    if Dv < D[v]:
                        for i, t in enumerate(L):
                            if t[1] is v:
                                L.pop(i)
                                break
                        D[v] = Dv
                        heappush(L, (Dv, v))
                else:
                    D[v] = Dv
                    heappush(L, (Dv, v))
        return D

    def get_scs_with_feedback(self, roots=None):
        from sys import getrecursionlimit, setrecursionlimit

        limit = getrecursionlimit()
        N = self.norm() + 10
        if N > limit:
            setrecursionlimit(N)

        def _visit(v, L):
            v.ind = v.ncur
            v.lowlink = v.ncur
            Vertex.ncur += 1
            self.tstack.append(v)
            v.mark = True
            for e in v.e_out():
                w = e.v[1]
                if w.ind == 0:
                    _visit(w, L)
                    v.lowlink = min(v.lowlink, w.lowlink)
                elif w.mark:
                    e.feedback = True
                if w in self.tstack:
                    v.lowlink = min(v.lowlink, w.ind)
            if v.lowlink == v.ind:
                l = [self.tstack.pop()]
                while l[0] != v:
                    l.insert(0, self.tstack.pop())
                # print "unstacked %s"%('-'.join([x.data[1:13] for x in l]))
                L.append(l)
            v.mark = False

        if roots is None:
            roots = self.roots()
        self.tstack = []
        scs = []
        Vertex.ncur = 1
        for v in self.sV:
            v.ind = 0
        # start exploring tree from roots:
        for v in roots:
            v = self.sV.get(v)
            if v.ind == 0:
                _visit(v, scs)
        # now possibly unvisited vertices:
        for v in self.sV:
            if v.ind == 0:
                _visit(v, scs)
        # clean up Tarjan-specific data:
        for v in self.sV:
            del v.ind
            del v.lowlink
            del v.mark
        del Vertex.ncur
        del self.tstack
        setrecursionlimit(limit)
        return scs

    def partition(self):
        V = self.sV.copy()
        R = self.roots()
        for r in R:
            V.remove(r)
        parts = []
        while len(R) > 0:
            v = R.pop(0)
            p = Poset([v])
            l = v.N(+1)
            while len(l) > 0:
                x = l.pop(0)
                if x in p:
                    continue
                if all([(y in p) for y in x.N(-1)]):
                    p.add(x)
                    if x in R:
                        R.remove(x)
                    else:
                        V.remove(x)
                    l.extend(x.N(+1))
                else:
                    if x in V:
                        V.remove(x)
                        R.append(x)
            parts.append(list(p))
        return parts

    def N(self, v, f_io=0):
        return v.N(f_io)

    # general graph properties:
    # -------------------------

    # returns True iff
    #  - o is a subgraph of self, or
    #  - o is a vertex in self, or
    #  - o is an edge in self
    def __contains__(self, o):
        try:
            return o.sV.issubset(self.sV) and o.sE.issubset(self.sE)
        except AttributeError:
            return (o in self.sV) or (o in self.sE)

    # merge GraphBase G into self
    def union_update(self, G):
        for v in G.sV:
            v.c = self
        self.sV.update(G.sV)
        self.sE.update(G.sE)

    # derivated graphs:
    # -----------------

    # returns subgraph spanned by vertices V
    def spans(self, V):
        raise NotImplementedError

    # returns join of G (if disjoint)
    def __mul__(self, G):
        raise NotImplementedError

    # returns complement of a graph G
    def complement(self, G):
        raise NotImplementedError

    # contraction G\e
    def contract(self, e):
        raise NotImplementedError

    def __getstate__(self):
        V = [v for v in self.sV]
        E = [e for e in self.sE]
        return (V, E, self.directed)

    def __setstate__(self, state):
        V, E, directed = state
        for e in E:
            e.v = [V[x] for x in e._v]
            del e._v
        GraphBase.__init__(self, V, E, directed)