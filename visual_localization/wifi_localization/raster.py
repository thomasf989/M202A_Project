"""Mesh depth rendering and exact sparse mesh ray lifting.

The original project used a small CPU rasterizer that dropped an entire
triangle whenever one vertex crossed the camera near plane.  That creates
artificial holes in the depth image.  This version clips triangles against the
near plane before rasterization and exposes bilinear depth sampling plus a
sparse ray-cast fallback for feature pixels near depth discontinuities.

The primary MeshLoc-style lifting path is still a rendered depth map sampled at
floating-point feature coordinates.  The ray fallback is only used when that
sample is unavailable or geometrically ambiguous.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _raster_triangle(image, a, b, c, K, width, height, near, far):
    # All three vertices must be in front of the near plane after clipping.
    if a[2] <= near or b[2] <= near or c[2] <= near:
        return
    ax, ay = a[0] / a[2] * K[0, 0] + K[0, 2], a[1] / a[2] * K[1, 1] + K[1, 2]
    bx, by = b[0] / b[2] * K[0, 0] + K[0, 2], b[1] / b[2] * K[1, 1] + K[1, 2]
    cx, cy = c[0] / c[2] * K[0, 0] + K[0, 2], c[1] / c[2] * K[1, 1] + K[1, 2]
    xmin = max(0, int(np.ceil(min(ax, bx, cx))))
    xmax = min(width - 1, int(np.floor(max(ax, bx, cx))))
    ymin = max(0, int(np.ceil(min(ay, by, cy))))
    ymax = min(height - 1, int(np.floor(max(ay, by, cy))))
    if xmin > xmax or ymin > ymax:
        return
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-10:
        return
    for y in range(ymin, ymax + 1):
        py = float(y)
        for x in range(xmin, xmax + 1):
            px = float(x)
            wa = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
            wb = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
            wc = 1.0 - wa - wb
            if min(wa, wb, wc) < -1e-7:
                continue
            # Perspective-correct interpolation of camera-space depth.
            invz = wa / a[2] + wb / b[2] + wc / c[2]
            if invz <= 0:
                continue
            z = 1.0 / invz
            if near < z < far and z < image[y, x]:
                image[y, x] = z


@njit(cache=True)
def _clip_triangle_near(a, b, c, near):
    """Clip a camera-space triangle against z > near.

    Returns a fixed-size polygon buffer and the number of valid vertices.
    A clipped triangle can become a triangle or a quad.
    """
    src = np.empty((5, 3), np.float64)
    dst = np.empty((5, 3), np.float64)
    src[0] = a
    src[1] = b
    src[2] = c
    nsrc = 3
    ndst = 0
    prev = src[nsrc - 1].copy()
    prev_in = prev[2] > near
    for i in range(nsrc):
        cur = src[i].copy()
        cur_in = cur[2] > near
        if cur_in != prev_in:
            denom = cur[2] - prev[2]
            if abs(denom) > 1e-12:
                t = (near - prev[2]) / denom
                p = prev + t * (cur - prev)
                # Put the generated point infinitesimally in front of the plane
                # so the rasterizer does not discard it numerically.
                p[2] = near + 1e-7
                dst[ndst] = p
                ndst += 1
        if cur_in:
            dst[ndst] = cur
            ndst += 1
        prev = cur
        prev_in = cur_in
    return dst, ndst


@njit(cache=True)
def _depth(vertices, faces, K, width, height, near, far):
    image = np.full((height, width), np.inf, np.float32)
    for face in faces:
        a = vertices[face[0]].astype(np.float64)
        b = vertices[face[1]].astype(np.float64)
        c = vertices[face[2]].astype(np.float64)
        if max(a[2], b[2], c[2]) <= near:
            continue
        poly, n = _clip_triangle_near(a, b, c, near)
        if n < 3:
            continue
        # Triangulate clipped triangle/quad as a fan.
        for i in range(1, n - 1):
            _raster_triangle(image, poly[0], poly[i], poly[i + 1], K,
                             width, height, near, far)
    return image


def mesh_depth(mesh, T_world_cv, K, width, height, near=.05, far=20.0):
    """Render camera-z depth for a mesh in an OpenCV camera grid."""
    T = np.asarray(T_world_cv, dtype=float)
    vertices = (np.asarray(mesh.vertices, dtype=float) - T[:3, 3]) @ T[:3, :3]
    return _depth(np.ascontiguousarray(vertices), np.asarray(mesh.faces, dtype=np.int64),
                  np.asarray(K, dtype=float), int(width), int(height), float(near), float(far))


def sample_depth_bilinear(depth, pixels, min_depth=.05, max_depth=20.0):
    """Bilinearly sample a depth image at floating-point pixel coordinates.

    This mirrors MeshLoc's depth lifting more closely than rounding keypoints to
    integer pixels.  ``spread`` reports the local 2x2 depth range and can be
    used to request an exact ray cast at discontinuities.
    """
    d = np.asarray(depth, dtype=float)
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    h, w = d.shape
    z = np.full(len(uv), np.nan, dtype=float)
    spread = np.full(len(uv), np.inf, dtype=float)
    valid = np.zeros(len(uv), dtype=bool)
    for i, (u, v) in enumerate(uv):
        if not (np.isfinite(u) and np.isfinite(v)) or u < 0 or v < 0 or u > w - 1 or v > h - 1:
            continue
        x0, y0 = int(np.floor(u)), int(np.floor(v))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        dx, dy = float(u - x0), float(v - y0)
        vals = np.array([d[y0, x0], d[y0, x1], d[y1, x0], d[y1, x1]], float)
        needed = []
        weights = []
        # Degenerate border cases reduce naturally to 1-D / nearest sampling.
        entries = [
            (vals[0], (1 - dx) * (1 - dy)),
            (vals[1], dx * (1 - dy)),
            (vals[2], (1 - dx) * dy),
            (vals[3], dx * dy),
        ]
        for value, weight in entries:
            if weight > 1e-12:
                needed.append(value)
                weights.append(weight)
        needed = np.asarray(needed, float)
        weights = np.asarray(weights, float)
        if not len(needed) or not np.isfinite(needed).all():
            continue
        zi = float(np.dot(needed, weights) / max(weights.sum(), 1e-12))
        if not (min_depth < zi < max_depth):
            continue
        z[i] = zi
        spread[i] = float(np.ptp(needed)) if len(needed) > 1 else 0.0
        valid[i] = True
    return z, valid, spread


def raycast_pixels(intersector, T_world_cv, K, pixels, min_depth=.05, max_depth=20.0):
    """Return exact world-space mesh intersections for sparse camera pixels."""
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    xyz = np.full((len(uv), 3), np.nan, dtype=float)
    valid = np.zeros(len(uv), dtype=bool)
    if not len(uv):
        return xyz, valid
    T = np.asarray(T_world_cv, dtype=float)
    K = np.asarray(K, dtype=float)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    dirs_cam = np.column_stack([x, y, np.ones(len(uv))])
    dirs_world = dirs_cam @ T[:3, :3].T
    dirs_world /= np.maximum(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-12)
    origins = np.repeat(T[None, :3, 3], len(uv), axis=0)
    locations, index_ray, _ = intersector.intersects_location(origins, dirs_world, multiple_hits=False)
    if len(index_ray):
        cam = (locations - T[:3, 3]) @ T[:3, :3]
        good = (cam[:, 2] > min_depth) & (cam[:, 2] < max_depth)
        for loc, ridx, ok in zip(locations, index_ray, good):
            if ok:
                xyz[int(ridx)] = loc
                valid[int(ridx)] = True
    return xyz, valid
