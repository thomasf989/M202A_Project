"""Perspective-correct mesh depth in the RGB camera grid, CPU only.

Numba compiles the triangle loop once. No OpenGL, display server or GPU is
required. Both triangle sides are rendered because room meshes are open.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _depth(vertices, faces, K, width, height, near, far):
    image = np.full((height, width), np.inf, np.float32)
    for face in faces:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Conservatively omit triangles crossing the camera near plane.
        if min(a[2], b[2], c[2]) <= near:
            continue
        ax, ay = a[0]/a[2]*K[0, 0]+K[0, 2], a[1]/a[2]*K[1, 1]+K[1, 2]
        bx, by = b[0]/b[2]*K[0, 0]+K[0, 2], b[1]/b[2]*K[1, 1]+K[1, 2]
        cx, cy = c[0]/c[2]*K[0, 0]+K[0, 2], c[1]/c[2]*K[1, 1]+K[1, 2]
        xmin = max(0, int(np.ceil(min(ax, bx, cx))))
        xmax = min(width-1, int(np.floor(max(ax, bx, cx))))
        ymin = max(0, int(np.ceil(min(ay, by, cy))))
        ymax = min(height-1, int(np.floor(max(ay, by, cy))))
        den = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den) < 1e-10:
            continue
        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):
                wa = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den
                wb = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den
                wc = 1-wa-wb
                if min(wa, wb, wc) < -1e-7:
                    continue
                z = 1/(wa/a[2]+wb/b[2]+wc/c[2])
                if near < z < far and z < image[y, x]:
                    image[y, x] = z
    return image


def mesh_depth(mesh, T_world_cv, K, width, height):
    T = np.asarray(T_world_cv)
    vertices = (mesh.vertices-T[:3, 3]) @ T[:3, :3]
    return _depth(np.ascontiguousarray(vertices), np.asarray(mesh.faces),
                  np.asarray(K), int(width), int(height), .05, 10.)
