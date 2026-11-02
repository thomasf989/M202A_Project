"""Phone photo decoding, explicit calibration, and approximate focal estimation."""
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares
from .polycam import camera_matrix
from .localization import pose_dict


def read_photo(path, max_edge=1600):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        if im.width*im.height > 65_000_000:
            raise ValueError('The photo is too large. Export it as a JPG with no more than 65 million pixels.')
        exif = im.getexif()
        extra = exif.get_ifd(34665) if 34665 in exif else {}
        focal35 = extra.get(41989, exif.get(41989))
        try:
            focal35 = float(focal35)
            if not 8 <= focal35 <= 500:
                focal35 = None
        except (ValueError, TypeError):
            focal35 = None
        oriented = ImageOps.exif_transpose(im).convert('RGB')
        width, height = oriented.size
        metadata = dict(width=width, height=height, focal_35mm=focal35,
            camera=str(exif.get(272, '')), exif_orientation=int(exif.get(274, 1)))
        oriented.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        rgb = np.asarray(oriented).copy()
    metadata['processing_width'], metadata['processing_height'] = rgb.shape[1], rgb.shape[0]
    return rgb, metadata


def validate_calibration(c):
    if not isinstance(c, dict):
        raise ValueError('The camera parameters file must be a JSON object.')
    for key in ['width', 'height', 'fx', 'fy', 'cx', 'cy']:
        if key not in c or not isinstance(c[key], (int, float)) or not np.isfinite(c[key]):
            raise ValueError(f'The camera parameters are missing a valid {key}.')
    if int(c['width']) != c['width'] or int(c['height']) != c['height'] or min(c['width'], c['height']) < 32:
        raise ValueError('Camera width and height must be valid pixel dimensions.')
    camera_matrix(c)
    d = np.array(c.get('distortion', [0, 0, 0, 0, 0]), dtype=float)
    if d.ndim != 1 or len(d) not in [4, 5, 8, 12, 14] or not np.isfinite(d).all():
        raise ValueError('The distortion parameters have an invalid format.')
    return {k: c[k] for k in ['width', 'height', 'fx', 'fy', 'cx', 'cy']} | {'distortion': d.tolist()}


def intrinsics_for_photo(metadata, mode='auto', calibration=None):
    w, h = metadata['processing_width'], metadata['processing_height']
    if mode == 'saved':
        if calibration is None:
            raise ValueError('No camera parameters have been saved. Import a JSON file or select automatic estimation.')
        c = validate_calibration(calibration)
        original_ratio = metadata['width']/metadata['height']
        if abs(original_ratio/(c['width']/c['height'])-1) > .005:
            raise ValueError('The photo orientation or aspect ratio differs from the saved camera parameters. Use matching parameters or select automatic estimation.')
        K = camera_matrix(c)
        sx, sy = w/c['width'], h/c['height']
        K[0, 0] *= sx;K[1, 1] *= sy
        K[0, 2] = (K[0, 2]+.5)*sx-.5
        K[1, 2] = (K[1, 2]+.5)*sy-.5
        return [K], np.array(c['distortion']), dict(source='saved', label='Saved camera parameters', estimated=False,
            note='Reuse only with the same lens, crop, and orientation.')
    if mode != 'auto':
        raise ValueError('Unknown camera parameter mode.')
    focal35 = metadata.get('focal_35mm')
    if focal35:
        # A 35mm-equivalent focal length refers to the full-frame diagonal.
        f = focal35*np.hypot(w, h)/np.hypot(36, 24)
        focals = [f*.82, f, f*1.18]
        label = f'Photo focal-length reference ({focal35:g} mm)'
    else:
        focals = np.array([.42, .52, .62, .72, .85, 1.00, 1.20, 1.45])*max(w, h)
        label = 'Estimated from image matching'
    candidates = [np.array([[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1]], float) for f in focals]
    return candidates, np.zeros(5), dict(source='exif' if focal35 else 'search', label=label, estimated=True,
        note='The camera parameters are estimates; import calibration parameters to verify localization accuracy.')


def refine_focal(result, image_size):
    """Jointly refine focal, R and t using only RANSAC's 2D-3D inliers.

    This does not turn an estimate into a camera calibration. Principal point,
    square pixels and zero distortion remain assumptions in automatic mode.
    """
    if not result.get('success') or result.get('inliers', 0) < 20:
        return result
    xyz = np.asarray(result['inlier_world_points'], float)
    xy = np.asarray(result['inlier_query_pixels'], float)
    K = np.array(result['query_intrinsics'], float)
    T = np.array(result['pose']['T_world_camera_cv'], float)
    R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    r = cv2.Rodrigues(R)[0].ravel()
    initial = np.r_[r, t, np.log(K[0, 0])]
    max_side = max(image_size)
    if not .30*max_side < K[0, 0] < 3*max_side:
        return result
    def residual(p):
        k = K.copy();k[0, 0] = k[1, 1] = np.exp(p[6])
        return (cv2.projectPoints(xyz, p[:3], p[3:6], k, None)[0].reshape(-1, 2)-xy).ravel()
    opt = least_squares(residual, initial, loss='soft_l1', f_scale=1., max_nfev=70,
        bounds=(np.r_[[-np.inf]*6, np.log(.30*max_side)], np.r_[[np.inf]*6, np.log(3*max_side)]))
    if not opt.success or np.linalg.norm(residual(opt.x)) > np.linalg.norm(residual(initial)):
        return result
    R = cv2.Rodrigues(opt.x[:3])[0];t = opt.x[3:6]
    if np.mean((xyz @ R.T + t)[:, 2] > .05) < .99:
        return result
    updated = np.eye(4);updated[:3, :3] = R.T;updated[:3, 3] = -R.T @ t
    K[0, 0] = K[1, 1] = float(np.exp(opt.x[6]))
    errors = np.linalg.norm(residual(opt.x).reshape(-1, 2), axis=1)
    result.update(pose=pose_dict(updated), query_intrinsics=K.tolist(),
        median_reprojection_px=float(np.median(errors)), p90_reprojection_px=float(np.percentile(errors, 90)),
        focal_refined=True)
    return result
