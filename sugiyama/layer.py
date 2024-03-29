from bisect import bisect


class Layer(list):
    """
    Layer is where Sugiyama layout organises vertices in hierarchical lists.
    The placement of a vertex is done by the Sugiyama class, but it highly relies on
    the *ordering* of vertices in each layer to reduce crossings.
    This ordering depends on the neighbors found in the upper or lower layers.

    Attributes:
        layout (SugiyamaLayout): a reference to the sugiyama layout instance that
                                 contains this layer
        upper (Layer): a reference to the *upper* layer (rank-1)
        lower (Layer): a reference to the *lower* layer (rank+1)
        ccount (int) : number of crossings detected in this layer

    Methods:
        setup (layout): set initial attributes values from provided layout
        nextlayer(): returns *next* layer in the current layout's direction parameter.
        prevlayer(): returns *previous* layer in the current layout's direction parameter.
        order(): compute *optimal* ordering of vertices within the layer.
    """

    __r = None
    layout = None
    upper = None
    lower = None
    __x = 1.0
    ccount = None

    def __eq__(self, other):
        return super().__eq__(other)

    def __str__(self):
        s = "<Layer %d" % self.__r
        s += ", len=%d" % len(self)
        xc = self.ccount or "?"
        s += ", crossings=%s>" % xc
        return s

    def setup(self, layout):
        self.layout = layout
        r = layout.layers.index(self)
        self.__r = r
        if len(self) > 1:
            self.__x = 1.0 / (len(self) - 1)
        for i, v in enumerate(self):
            assert layout.grx[v].rank == r
            layout.grx[v].pos = i
            layout.grx[v].bar = i * self.__x
        if r > 0:
            self.upper = layout.layers[r - 1]
        if r < len(layout.layers) - 1:
            self.lower = layout.layers[r + 1]

    def nextlayer(self):
        return self.lower if self.layout.dirv == -1 else self.upper

    def prevlayer(self):
        return self.lower if self.layout.dirv == +1 else self.upper

    def order(self):
        sug = self.layout
        sug._edge_inverter()
        c = self._cc()
        if c > 0:
            for v in self:
                sug.grx[v].bar = self._meanvalueattr(v)
            # now resort layers l according to bar value:
            self.sort(key=lambda x: sug.grx[x].bar)
            # reduce & count crossings:
            c = self._ordering_reduce_crossings()
            # assign new position in layer l:
            for i, v in enumerate(self):
                sug.grx[v].pos = i
                sug.grx[v].bar = i * self.__x
        sug._edge_inverter()
        self.ccount = c
        return c

    def _meanvalueattr(self, v):
        """
        find new position of vertex v according to adjacency in prevlayer.
        position is given by the mean value of adjacent positions.
        experiments show that meanvalue heuristic performs better than median.
        """
        sug = self.layout
        if not self.prevlayer():
            return sug.grx[v].bar
        bars = [sug.grx[x].bar for x in self._neighbors(v)]
        return sug.grx[v].bar if len(bars) == 0 else float(sum(bars)) / len(bars)

    def _medianindex(self, v):
        """
        find new position of vertex v according to adjacency in layer l+dir.
        position is given by the median value of adjacent positions.
        median heuristic is proven to achieve at most 3 times the minimum
        of crossings (while barycenter achieve in theory the order of |V|)
        """
        assert self.prevlayer() != None
        N = self._neighbors(v)
        g = self.layout.grx
        pos = [g[x].pos for x in N]
        lp = len(pos)
        if lp == 0:
            return []
        pos.sort()
        pos = pos[:: self.layout.dirh]
        i, j = divmod(lp - 1, 2)
        return [pos[i]] if j == 0 else [pos[i], pos[i + j]]

    def _neighbors(self, v):
        """
        neighbors refer to upper/lower adjacent nodes.
        Note that v.neighbors() provides neighbors of v in the graph, while
        this method provides the Vertex and DummyVertex adjacent to v in the
        upper or lower layer (depending on layout.dirv state).
        """
        assert self.layout.dag
        dirv = self.layout.dirv
        grxv = self.layout.grx[v]
        try:  # (cache)
            return grxv.nvs[dirv]
        except AttributeError:
            grxv.nvs = {-1: v.neighbors(-1), +1: v.neighbors(+1)}
            if grxv.dummy:
                return grxv.nvs[dirv]
            # v is real, v.neighbors are graph neigbors but we need layers neighbors
            for d in (-1, +1):
                tr = grxv.rank + d
                for i, x in enumerate(v.neighbors(d)):
                    if self.layout.grx[x].rank == tr:
                        continue
                    e = v.e_with(x)
                    dum = self.layout.ctrls[e][tr]
                    grxv.nvs[d][i] = dum
            return grxv.nvs[dirv]

    def _crossings(self):
        """
        counts (inefficently but at least accurately) the number of
        crossing edges between layer l and l+dirv.
        P[i][j] counts the number of crossings from j-th edge of vertex i.
        The total count of crossings is the sum of flattened P:
        x = sum(sum(P,[]))
        """
        g = self.layout.grx
        P = []
        for v in self:
            P.append([g[x].pos for x in self._neighbors(v)])
        for i, p in enumerate(P):
            candidates = sum(P[i + 1 :], [])
            for j, e in enumerate(p):
                p[j] = len(filter((lambda nx: nx < e), candidates))
            del candidates
        return P

    def _cc(self):
        """
        implementation of the efficient bilayer cross counting by insert-sort
        (see Barth & Mutzel paper "Simple and Efficient Bilayer Cross Counting")
        """
        g = self.layout.grx
        P = []
        for v in self:
            P.extend(sorted([g[x].pos for x in self._neighbors(v)]))
        # count inversions in P:
        s = []
        count = 0
        for i, p in enumerate(P):
            j = bisect(s, p)
            if j < i:
                count += i - j
            s.insert(j, p)
        return count

    def _ordering_reduce_crossings(self):
        assert self.layout.dag
        g = self.layout.grx
        N = len(self)
        X = 0
        for i, j in zip(range(N - 1), range(1, N)):
            vi = self[i]
            vj = self[j]
            ni = [g[v].bar for v in self._neighbors(vi)]
            Xij = Xji = 0
            for nj in [g[v].bar for v in self._neighbors(vj)]:
                x = len([nx for nx in ni if nx > nj])
                Xij += x
                Xji += len(ni) - x
            if Xji < Xij:
                self[i] = vj
                self[j] = vi
                X += Xji
            else:
                X += Xij
        return X