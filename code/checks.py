#!/usr/bin/env python3
"""Exact checks for "The largest box in a sparse random allocation".

All comparisons are made in exact rational arithmetic (fractions.Fraction); floating point is used
only to print the largest ratio of a probability to its bound.  Python 3.8 or later, standard
library only.  Run:  python checks.py

Part 1  Lemma 2.1: a, the smallest elements c of the residue classes of M, C and gamma, and the
        representation of every j in M with at least (j - C)/a parts, each at most max(a, C).
Part 2  Lemma 3.2 by brute force over all allocations of small size: the weight identity, the
        number of images and preimages, and the summed inequality.
Part 3  Theorem 1.1 for eight weight sequences, every n <= NMAX and every 1 <= m <= gamma n/4 with
        Z(m, n) > 0, and for larger n: inequality (3.3), inequality (1.2) for every j, and the tail
        bound (1.3) for every k >= max(1, C).
Part 4  Lemma 5.1 (uniform weights): the product formula, negative correlation, and the second
        moment bound P(Y(1) < k) <= 1/(n P(Y_1 >= k)).
Part 5  Negative controls: strengthened forms of the inequalities, each of which must fail.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from math import comb, factorial
import sys

NMAX = 64
LARGE = [(n, m) for n in (100, 200, 400) for m in (1, 2, 3, 5, 10, 20)]
KMAX = 80

# name, weight function, r with 0 < r < rho, a rational upper bound for Phi(r)
FAMILIES = [
    ("Maxwell-Boltzmann w_k = 1/k!", lambda k: F(1, factorial(k)), F(1), F(3)),
    ("Bose-Einstein w_k = 1", lambda k: F(1), F(1, 2), F(2)),
    ("geometric w_k = 2^-k", lambda k: F(1, 2 ** k), F(1), F(2)),
    ("linear w_k = k+1", lambda k: F(k + 1), F(1, 2), F(4)),
    ("heavy empty box w_0 = 3, w_k = 1", lambda k: F(3) if k == 0 else F(1), F(1, 2), F(4)),
    ("no singletons w_1 = 0, w_k = 1", lambda k: F(0) if k == 1 else F(1), F(1, 2), F(3, 2)),
    ("support {0,3,5}", lambda k: F(1) if k in (0, 3, 5) else F(0), F(1), F(3)),
    ("support {0,2,4}, w_4 = 1/2", lambda k: {0: F(1), 2: F(1), 4: F(1, 2)}.get(k, F(0)), F(1), F(5, 2)),
]

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    if len(FAILURES) <= 10:
        print("  FAILURE:", msg)


# ---------------------------------------------------------------- arithmetic of the support
def support_data(w):
    D = [k for k in range(1, KMAX + 1) if w(k) > 0]
    a = D[0]
    rep = {0: ()}
    for x in range(1, KMAX + 1):
        for d in D:
            if d > x:
                break
            if x - d in rep:
                rep[x] = tuple(sorted(rep[x - d] + (d,)))
                break
    c = {}
    for x in sorted(rep):
        c.setdefault(x % a, x)
    C = max(c.values())
    assert C + a < KMAX // 2, "residue classes not resolved below KMAX"
    gamma = min([F(1)] + [w(f) / w(0) for f in D if f <= max(a, C)])

    def rep_j(j):
        cj = c[j % a]
        return tuple(sorted(rep[cj] + (a,) * ((j - cj) // a)))

    return D, a, c, C, gamma, rep, rep_j


def iroot(v, a):
    """floor(v ** (1/a)) for an integer v >= 0."""
    if v < 2:
        return v
    x = 1 << ((v.bit_length() + a - 1) // a)
    while True:
        y = ((a - 1) * x + v // x ** (a - 1)) // a
        if y >= x:
            break
        x = y
    while x ** a > v:
        x -= 1
    while (x + 1) ** a <= v:
        x += 1
    return x


def root_lower(x, a, S=10 ** 15):
    """A rational l <= x ** (1/a), within 1/S of it."""
    return F(iroot(x.numerator * S ** a // x.denominator, a), S)


def root_upper(x, a, S=10 ** 15):
    """A rational u >= x ** (1/a), within 2/S of it."""
    return root_lower(x, a, S) + F(1, S)


# ---------------------------------------------------------------- power series
def mul(p, q, d):
    out = [F(0)] * (d + 1)
    for i, x in enumerate(p[:d + 1]):
        if x:
            for j in range(d + 1 - i):
                y = q[j]
                if y:
                    out[i + j] += x * y
    return out


def power(c, n, d):
    res = [F(1)] + [F(0)] * d
    base = (list(c) + [F(0)] * (d + 1))[:d + 1]
    while n:
        if n & 1:
            res = mul(res, base, d)
        n >>= 1
        if n:
            base = mul(base, base, d)
    return res


def allocations(m, n):
    if n == 1:
        yield (m,)
        return
    for first in range(m + 1):
        for rest in allocations(m - first, n - 1):
            yield (first,) + rest


def weight(y, w):
    out = F(1)
    for v in y:
        out *= w(v)
    return out


# ---------------------------------------------------------------- Part 1
def part1():
    print("Part 1: Lemma 2.1 (the arithmetic of the support)")
    total = 0
    for name, w, r, phi in FAMILIES:
        D, a, c, C, gamma, rep, rep_j = support_data(w)
        for j in range(1, KMAX - a):
            in_M = j in rep
            via_classes = (j % a) in c and c[j % a] <= j
            total += 1
            if in_M != via_classes:
                fail("%s: membership of %d" % (name, j))
                continue
            if not in_M:
                continue
            parts = rep_j(j)
            total += 1
            if sum(parts) != j or any(g not in D or g > max(a, C) for g in parts) or a * len(parts) < j - C:
                fail("%s: representation of %d" % (name, j))
        print("  %-34s a = %d, C = %2d, gamma = %s, classes %s" % (
            name, a, C, gamma, ", ".join("%d:%d" % (k, v) for k, v in sorted(c.items()))))
    print("  total: %d checks" % total)


# ---------------------------------------------------------------- Part 2
def part2():
    print("Part 2: Lemma 3.2 by brute force (all allocations, n <= 7, m <= 6)")
    total = 0
    for name, w, r, phi in FAMILIES:
        D, a, c, C, gamma, rep, rep_j = support_data(w)
        w0 = w(0)
        for n in range(2, 8):
            for m in range(1, 7):
                Zmn = sum(weight(z, w) for z in allocations(m, n))
                if Zmn == 0:
                    continue
                for j in range(1, m + 1):
                    if j not in rep:
                        continue
                    parts = rep_j(j)
                    p = len(parts)
                    if p > n - 1 - m:
                        continue
                    factor = w0
                    for g in parts:
                        factor *= w(g) / w0
                    lhs_sum = F(0)
                    pre = Counter()
                    for y in allocations(m - j, n - 1):
                        empty = [i for i in range(n - 1) if y[i] == 0]
                        zs = []
                        for J in combinations(empty, p):
                            z = list(y) + [0]
                            for s, i in enumerate(J):
                                z[i] = parts[s]
                            zs.append(tuple(z))
                        total += 3
                        if len(zs) != comb(len(empty), p) or len(set(zs)) != len(zs):
                            fail("%s n=%d m=%d j=%d: images of %s" % (name, n, m, j, y))
                        if len(empty) < n - 1 - m:
                            fail("%s n=%d m=%d j=%d: empty boxes of %s" % (name, n, m, j, y))
                        wy = weight(y, w)
                        for z in zs:
                            if weight(z, w) != wy * factor:
                                fail("%s: weight identity at %s" % (name, z))
                            pre[z] += 1
                            lhs_sum += weight(z, w)
                    for z, cnt in pre.items():
                        total += 1
                        if cnt > comb(sum(1 for v in z if v), p) or cnt > comb(m, p):
                            fail("%s n=%d m=%d j=%d: preimages of %s" % (name, n, m, j, z))
                    Zy = sum(weight(y, w) for y in allocations(m - j, n - 1))
                    total += 2
                    if not comb(n - 1 - m, p) * w0 * gamma ** p * Zy <= lhs_sum <= comb(m, p) * Zmn:
                        fail("%s n=%d m=%d j=%d: summed inequality" % (name, n, m, j))
                    if not comb(n - 1 - m, p) * w0 * gamma ** p * Zy <= comb(m, p) * Zmn:
                        fail("%s n=%d m=%d j=%d: (3.2)" % (name, n, m, j))
    print("  total: %d checks" % total)


# ---------------------------------------------------------------- Part 3
class Tally:
    def __init__(self):
        self.points = self.b33 = self.b12 = self.tail = 0
        self.ratio = 0.0
        self.ctrl_exp = self.ctrl_n = 0          # control failures found (expected)
        self.ctrl_exp_tried = self.ctrl_n_tried = 0


def check_point(fam, sd, n, m, Zn, Zn1, trunc, t):
    name, w, r, phi = fam
    D, a, c, C, gamma, rep, rep_j = sd
    w0 = w(0)
    x = F(2 * m) / (gamma * n)
    t.points += 1
    for j in range(1, m + 1):
        P = w(j) * Zn1[m - j] / Zn[m]
        if w(j) == 0:
            if P != 0:
                fail("%s n=%d m=%d j=%d: zero weight, positive probability" % (name, n, m, j))
            continue
        p = len(rep_j(j))
        b = w(j) / w0
        t.b33 += 1
        if not P <= b * x ** p:
            fail("%s n=%d m=%d j=%d: (3.3)" % (name, n, m, j))
        t.b12 += 1
        if not P ** a <= b ** a * x ** (j - C):
            fail("%s n=%d m=%d j=%d: (1.2)" % (name, n, m, j))
        if P > 0:
            t.ratio = max(t.ratio, float(P) / (float(b) * float(x) ** ((j - C) / a)))
        t.ctrl_exp_tried += 1
        if P > b * x ** (p + 1):
            t.ctrl_exp += 1
    if not x < r ** a:
        return
    # (1.3) in a stronger form: Phi(r) is replaced by max_{1 <= j <= m} w_j r^j (P(Y_1 = j) = 0 for
    # j > m), and x^(1/a) by a rational lower bound lo, which lowers the right side when k >= C
    A = max(w(j) * r ** j for j in range(1, m + 1)) / w0
    lo, up = root_lower(x, a), root_upper(x, a)
    for k in range(max(1, C), m + 1):
        tail = 1 - trunc[k] / Zn[m]
        t.tail += 1
        if not tail <= A * n * r ** (-k) * lo ** (k - C) / (1 - lo / r):
            fail("%s n=%d m=%d k=%d: (1.3)" % (name, n, m, k))
        if up < r:
            t.ctrl_n_tried += 1
            if tail > (phi / w0) * r ** (-k) * up ** (k - C) / (1 - up / r):
                t.ctrl_n += 1


def part3():
    print("Part 3: Theorem 1.1, exactly; n <= %d and every 1 <= m <= gamma n/4, then n = 100, 200, 400" % NMAX)
    tallies = []
    for fam in FAMILIES:
        name, w, r, phi = fam
        sd = support_data(w)
        gamma = sd[4]
        t = Tally()
        dmax = int(gamma * NMAX / 4)
        phi_c = [w(k) for k in range(dmax + 1)]
        trunc_c = {k: [w(i) if i < k else F(0) for i in range(dmax + 1)] for k in range(1, dmax + 1)}
        Zprev = [F(1)] + [F(0)] * dmax
        Tprev = {k: [F(1)] + [F(0)] * dmax for k in trunc_c}
        for n in range(1, NMAX + 1):
            Zn = mul(Zprev, phi_c, dmax)
            Tn = {k: mul(Tprev[k], trunc_c[k], dmax) for k in trunc_c}
            if n >= 4:
                for m in range(1, dmax + 1):
                    if 4 * m <= gamma * n and Zn[m] > 0:
                        check_point(fam, sd, n, m, Zn, Zprev, {k: Tn[k][m] for k in Tn}, t)
            Zprev, Tprev = Zn, Tn
        small = (t.points, t.b12, t.tail)
        for n, m in LARGE:
            if not 4 * m <= gamma * n:
                continue
            d = m
            phi_c = [w(k) for k in range(d + 1)]
            Zn1 = power(phi_c, n - 1, d)
            Zn = mul(Zn1, phi_c, d)
            if Zn[m] == 0:
                continue
            trunc = {k: power([w(i) if i < k else F(0) for i in range(d + 1)], n, d)[m] for k in range(1, m + 1)}
            check_point(fam, sd, n, m, Zn, Zn1, trunc, t)
        print("  %-34s %4d points (%d with n > %d): %4d of (1.2), %4d of (1.3); largest P(Y_1 = j)/bound %.4f"
              % (name, t.points, t.points - small[0], NMAX, t.b12, t.tail, t.ratio))
        tallies.append(t)
    tot = sum(t.b33 + t.b12 + t.tail for t in tallies)
    print("  total: %d checks" % tot)
    return tallies


# ---------------------------------------------------------------- Part 4
def part4():
    print("Part 4: Lemma 5.1 (uniform weights)")
    total = 0

    def tail1(m, n, k):
        out = F(1)
        for i in range(k):
            out *= F(m - i, m + n - 1 - i)
        return out

    # the product formulas against a full enumeration
    for n in range(2, 7):
        for m in range(1, 11):
            allocs = list(allocations(m, n))
            for k in range(1, m + 1):
                one = F(sum(1 for y in allocs if y[0] >= k), len(allocs))
                two = F(sum(1 for y in allocs if y[0] >= k and y[1] >= k), len(allocs))
                total += 2
                if one != tail1(m, n, k):
                    fail("product formula n=%d m=%d k=%d" % (n, m, k))
                if two != (tail1(m, n, 2 * k) if 2 * k <= m else 0):
                    fail("joint product formula n=%d m=%d k=%d" % (n, m, k))
    # negative correlation and the second moment bound
    d = 40
    ones = [F(1)] * (d + 1)
    for n in range(2, 121):
        for m in range(1, d + 1):
            for k in range(1, m + 1):
                total += 1
                if 2 * k <= m and not tail1(m, n, 2 * k) <= tail1(m, n, k) ** 2:
                    fail("negative correlation n=%d m=%d k=%d" % (n, m, k))
    for n in range(2, 61):
        Zn = power(ones, n, d)
        for k in range(1, d + 1):
            Tk = power([F(1) if i < k else F(0) for i in range(d + 1)], n, d)
            for m in range(k, d + 1):
                below = Tk[m] / Zn[m]
                total += 1
                if not below <= 1 / (n * tail1(m, n, k)):
                    fail("second moment n=%d m=%d k=%d" % (n, m, k))
    print("  total: %d checks" % total)


# ---------------------------------------------------------------- Part 5
def part5(tallies):
    print("Part 5: negative controls (each must fail somewhere)")
    ok = True
    e_f = sum(t.ctrl_exp for t in tallies)
    e_t = sum(t.ctrl_exp_tried for t in tallies)
    print("  (3.3) with one more factor x:                  fails in %d of %d cases" % (e_f, e_t))
    ok &= e_f > 0
    n_f = sum(t.ctrl_n for t in tallies)
    n_t = sum(t.ctrl_n_tried for t in tallies)
    print("  (1.3) without the factor n:                    fails in %d of %d cases" % (n_f, n_t))
    ok &= n_f > 0
    # the additive constant C: w_1 = 0 < w_2 w_3 and m = 3 force Y(1) = 3, so (1.2) with j/a in
    # place of (j - C)/a fails once n is large
    name, w, r, phi = FAMILIES[5]
    _, a, _, C, gamma, _, _ = support_data(w)
    hits = 0
    for n in (12, 100, 1000, 10000):
        m = 3
        d = m
        phi_c = [w(k) for k in range(d + 1)]
        Zn1 = power(phi_c, n - 1, d)
        Zn = mul(Zn1, phi_c, d)
        P3 = w(3) * Zn1[0] / Zn[3]
        x = F(2 * m) / (gamma * n)
        if P3 ** a > (w(3) / w(0)) ** a * x ** 3:
            hits += 1
        if P3 != F(1, n):
            fail("w_1 = 0, m = 3: P(Y_1 = 3) = %s at n = %d" % (P3, n))
    print("  (1.2) with j/a in place of (j - C)/a, w_1 = 0, m = 3, n = 12, 100, 1000, 10000: fails for %d of 4" % hits)
    ok &= hits > 0
    if not ok:
        fail("a negative control did not fail")


def main():
    part1()
    part2()
    tallies = part3()
    part4()
    part5(tallies)
    if FAILURES:
        print("FAILURES: %d" % len(FAILURES))
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
