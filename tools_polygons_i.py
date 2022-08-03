import shapely.geometry as geom
from copy import copy
import math
import munkres
import scipy.spatial as spatial
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
def as_tuple(point):return (point.xy[0][0], point.xy[1][0])
# ----------------------------------------------------------------------------------------------------------------------
class PolygonInterpolator:
    def __init__(self, p1=None, p2=None):
        if p1 is not None and p2 is not None:
            self.p1 = p1
            self.p2 = p2

            self.compute_interpolation()
            self.compute_vertex_order()
        else:
            pass

    # ----------------------------------------------------------------------------------------------------------------------
    def compute_interpolation(self):
        done = set([])

        pstrt = [geom.Point(p) for p in self.p1.exterior.coords[:-1]]
        pdest = [geom.Point(p) for p in self.p2.exterior.coords[:-1]]

        self.pairs, self.tuple_pairs = [], []

        while len(pdest) > len(pstrt):
            pstrt += self.midpoints(pstrt)

        mat = [[p1.distance(p2) for p2 in pstrt] for p1 in pdest]
        m = munkres.Munkres()
        indexes = m.compute(mat)

        for i in indexes:
            closest = pstrt[i[1]]
            p = pdest[i[0]]
            done.add((closest.x, closest.y))

            self.pairs.append((closest, p))
            self.tuple_pairs.append((as_tuple(closest), as_tuple(p)))

        ppp = done.symmetric_difference([(p.x, p.y) for p in pstrt])
        ppp = [geom.Point(p) for p in ppp]

        for p in ppp:
            closest, _ = self.project_point_points(p, pdest, 0)
            self.pairs.append((p, closest))
            self.tuple_pairs.append((as_tuple(p), as_tuple(closest)))

# ----------------------------------------------------------------------------------------------------------------------
    def compute_vertex_order(self):
        points = np.vstack(self.fast_interpolate_pairs(0.5))
        hull = spatial.ConvexHull(points)
        self.order = hull.vertices
        self.tuple_pairs = [self.tuple_pairs[i] for i in self.order]
        self.pairs = [self.pairs[i] for i in self.order]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def fast_interpolate_pairs(self, percent):
        perc = max(min(percent, 1.), -1.)
        if perc < 0:
            perc = 1 + perc
        return [((t1[0] * (1 - perc) + t2[0] * perc,t1[1] * (1 - perc) + t2[1] * perc)) for t1, t2 in self.tuple_pairs]

# ----------------------------------------------------------------------------------------------------------------------
    def midpoints(self, points):
        l = []
        for i, point in enumerate(points):
            prev = points[i - 1]
            mid = geom.Point((point.xy[0][0] + prev.xy[0][0]) / 2,(point.xy[1][0] + prev.xy[1][0]) / 2)
            l.append(mid)
        return l
# ----------------------------------------------------------------------------------------------------------------------
    def project_point_points(self,point, points, percent):
        dists = [point.distance(p) for p in points]
        dmin = min(dists)
        imin = dists.index(dmin)
        closest = points[imin]
        if imin < len(points) - 1:
            next_p = points[imin + 1]
        else:
            next_p = points[-1]

        prev_p = points[imin - 1]

        if point.distance(prev_p) > point.distance(next_p):
            edge = geom.LineString([as_tuple(closest), as_tuple(next_p)])
        else:
            edge = geom.LineString([as_tuple(prev_p), as_tuple(closest)])

        if point.xy == closest.xy:
            return closest, copy(closest)
        else:
            dist = edge.project(point)
            projection = edge.interpolate(dist)
            line = geom.LineString((as_tuple(point), as_tuple(projection)))
            interp = line.interpolate(percent, normalized=True)
            return projection, interp
# ----------------------------------------------------------------------------------------------------------------------
