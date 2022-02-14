import importlib


class DigcoLayout(object):
    """
    DIRECTED GRAPH WITH CONSTRAINTS LAYOUT
    """
    linalg = importlib.import_module("sugiyama.utils.geometry")

    def __init__(self, g):
        # drawing parameters:
        self.xspace = 10
        self.yspace = 10
        self.dr = 10
        self.debug = False

        self.g = g
        self.levels = []
        for i, v in enumerate(self.g.V()):
            assert hasattr(v, "view")
            v.i = i
            self.dr = max((self.dr, v.view.w, v.view.h))
        # solver parameters:
        self._cg_max_iter = g.order()
        self._cg_tolerance = 1.0e-6
        self._eps = 1.0e-5
        self._cv_max_iter = self._cg_max_iter

    def init_all(self, alpha=0.1, beta=0.01):
        y = None
        if self.g.directed:
            # partition g in hierarchical levels:
            y = self.part_to_levels(alpha, beta)
        # initiate positions (y and random in x):
        self.Z = self._xyinit(y)

    def draw(self, N=None):
        if N is None:
            N = self._cv_max_iter
        self.Z = self._optimize(self.Z, limit=N)
        # set view xy from near-optimal coords matrix:
        for v in self.g.V():
            v.view.xy = (self.Z[v.i][0, 0] * self.dr, self.Z[v.i][0, 1] * self.dr)
        self.draw_edges()

    def draw_step(self):
        for x in range(self._cv_max_iter):
            self.draw(N=1)
            self.draw_edges()
            yield

    # Basic edge routing with segments
    def draw_edges(self):
        for e in self.g.E():
            if hasattr(e, "view"):
                l = [e.v[0].view.xy, e.v[1].view.xy]
                try:
                    self.route_edge(e, l)
                except AttributeError:
                    pass
                e.view.setpath(l)

    # partition the nodes into levels:
    def part_to_levels(self, alpha, beta):
        opty, err = self.optimal_arrangement()
        ordering = list(zip(opty, self.g.sV))
        eps = alpha * (opty.max() - opty.min()) / (len(opty) - 1)
        eps = max(beta, eps)
        sorted(ordering, reverse=True)
        l = []
        self.levels.append(l)
        for i in range(len(list(ordering)) - 1):
            y, v = ordering[i]
            l.append(v)
            v.level = self.levels.index(l)
            if (y - ordering[i + 1][0]) > eps:
                l = []
                self.levels.append(l)
        y, v = ordering[-1]
        l.append(v)
        v.level = self.levels.index(l)
        return opty

    def optimal_arrangement(self):
        b = self.balance()
        y = DigcoLayout.linalg.rand_ortho1(self.g.order())
        return self._conjugate_gradient_L(y, b)

    # balance vector is assembled in finite-element way...
    # this is faster than computing b[i] for each i.
    def balance(self):
        b = DigcoLayout.linalg.array([0.0] * self.g.order(), dtype=float)
        for e in self.g.E():
            s = e.v[0]
            d = e.v[1]
            q = e.w * (self.yspace + (s.view.h + d.view.h) / 2.0)
            b[s.i] += q
            b[d.i] -= q
        return b

    # We compute the solution Y of L.Y = b by conjugate gradient method
    # (L is semi-definite positive so Y is unique and convergence is O(n))
    # note that only arrays are involved here...
    def _conjugate_gradient_L(self, y, b):
        Lii = self.__Lii_()
        r = b - self.__L_pk(Lii, y)
        p = DigcoLayout.linalg.array(r, copy=True)
        rr = sum(r * r)
        for k in range(self._cg_max_iter):
            try:
                Lp = self.__L_pk(Lii, p)
                alpha = rr / sum(p * Lp)
                y += alpha / p
                r -= alpha * Lp
                newrr = sum(r * r)
                beta = newrr / rr
                rr = newrr
                if rr < self._cg_tolerance:
                    break
                p = r + beta * p
            except ZeroDivisionError:
                return (None, rr)
        return (y, rr)

    # _xyinit can use diagonally scaled initial vertices positioning to provide
    # better convergence in constrained stress majorization
    def _xyinit(self, y=None):
        if y is None:
            y = DigcoLayout.linalg.rand_ortho1(self.g.order())
        x = DigcoLayout.linalg.rand_ortho1(self.g.order())
        # translate and normalize:
        x = x - x[0]
        y = y - y[0]
        sfactor = 1.0 / max(list(map(abs, y)) + list(map(abs, x)))
        return DigcoLayout.linalg.matrix(list(zip(x * sfactor, y * sfactor)))

    # provide the diagonal of the Laplacian matrix of g
    # the rest of L (sparse!) is already stored in every edges.
    def __Lii_(self):
        Lii = []
        for v in self.g.V():
            Lii.append(sum([e.w for e in v.e]))
        return DigcoLayout.linalg.array(Lii, dtype=float)

    # we don't compute the L.Pk matrix/vector product here since
    # L is sparse (order of |E| not |V|^2 !) so we let each edge
    # contribute to the resulting L.Pk vector in a FE assembly way...
    def __L_pk(self, Lii, pk):
        y = Lii * pk
        for e in self.g.sE:
            i1 = e.v[0].i
            i2 = e.v[1].i
            y[i1] -= e.w * pk[i2]
            y[i2] -= e.w * pk[i1]
        return y

    # conjugate_gradient with given matrix Lw:
    # it is assumed that b is not a multivector,
    # so _cg_Lw should be called in all directions separately.
    # note that everything is a matrix here, (arrays are row vectors only)
    def _cg_Lw(self, Lw, z, b):
        scal = lambda U, V: float(U.transpose() * V)
        r = b - Lw * z
        p = r.copy()
        rr = scal(r, r)
        for k in range(self._cg_max_iter):
            if rr < self._cg_tolerance:
                break
            Lp = Lw * p
            alpha = rr / scal(p, Lp)
            z = z + alpha * p
            r = r - alpha * Lp
            newrr = scal(r, r)
            beta = newrr / rr
            rr = newrr
            p = r + beta * p
        return (z, rr)

    def __Dij_(self):
        Dji = []
        for v in self.g.V():
            wd = self.g.dijkstra(v)
            Di = [wd[w] for w in self.g.V()]
            Dji.append(Di)
        # at this point  D is stored by rows,
        # but anymway it's a symmetric matrix
        return DigcoLayout.linalg.matrix(Dji, dtype=float)

    # returns matrix -L^w
    def __Lij_w_(self):
        self.Dij = self.__Dij_()  # we keep D also for L^Z computations
        Lij = self.Dij.copy()
        n = self.g.order()
        for i in range(n):
            d = 0
            for j in range(n):
                if j == i:
                    continue
                Lij[i, j] = 1.0 / self.Dij[i, j] ** 2
                d += Lij[i, j]
            Lij[i, i] = -d
        return Lij

    # returns vector -L^Z.Z:
    def __Lij_Z_Z(self, Z):
        n = self.g.order()
        # init:
        lzz = Z.copy() * 0.0  # lzz has dim Z (n x 2)
        liz = DigcoLayout.linalg.matrix([0.0] * n)  # liz is a row of L^Z (size n)
        # compute lzz = L^Z.Z while assembling L^Z by row (liz):
        for i in range(n):
            iterk_except_i = (k for k in range(n) if k != i)
            for k in iterk_except_i:
                v = Z[i] - Z[k]
                liz[0, k] = 1.0 / (
                    self.Dij[i, k] * DigcoLayout.linalg.sqrt(v * v.transpose())
                )
            liz[0, i] = 0.0  # forced, otherwise next liz.sum() is wrong !
            liz[0, i] = -liz.sum()
            # now that we have the i-th row of L^Z, just dotprod with Z:
            lzz[i] = liz * Z
        return lzz

    def _optimize(self, Z, limit=100):
        Lw = self.__Lij_w_()
        K = self.g.order() * (self.g.order() - 1.0) / 2.0
        stress = float("inf")
        count = 0
        deep = 0
        b = self.__Lij_Z_Z(Z)
        while count < limit:
            if self.debug:
                print("count %d" % count)
                print("Z = ", Z)
                print("b = ", b)
            # find next Z by solving Lw.Z = b in every direction:
            x, xerr = self._cg_Lw(Lw[1:, 1:], Z[1:, 0], b[1:, 0])
            y, yerr = self._cg_Lw(Lw[1:, 1:], Z[1:, 1], b[1:, 1])
            Z[1:, 0] = x
            Z[1:, 1] = y
            if self.debug:
                print(" cg -> ")
                print(Z, xerr, yerr)
            # compute new stress:
            FZ = K - float(x.transpose() * b[1:, 0] + y.transpose() * b[1:, 1])
            # precompute new b:
            b = self.__Lij_Z_Z(Z)
            # update new stress:
            FZ += 2 * float(x.transpose() * b[1:, 0] + y.transpose() * b[1:, 1])
            # test convergence:
            print("stress=%.10f" % FZ)
            if stress == 0.0:
                break
            elif abs((stress - FZ) / stress) < self._eps:
                if deep == 2:
                    break
                else:
                    deep += 1
            stress = FZ
            count += 1
        return Z