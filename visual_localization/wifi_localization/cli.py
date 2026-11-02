"""Run with python -m wifi_localization --help."""
import argparse
import csv
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .polycam import prepare_capture, read_json, write_json, rgb_image, query_calibration, sha256
from .capture import load_capture
from .features import build_database, Database
from .localization import Settings, localize, pose_dict, pose_errors
from .visualization import audit, scene_html, evaluation_plot


def settings(args):
    if args.top_k < 1 or not 0 < args.ratio < 1 or args.reprojection_px <= 0 or args.min_inliers < 6:
        raise ValueError('top-k >= 1, ratio in (0,1), reprojection-px > 0, min-inliers >= 6 required')
    return Settings(top_k=args.top_k, ratio=args.ratio, reprojection_px=args.reprojection_px,
                    min_inliers=args.min_inliers, seed=args.seed)


def capture_for_database(args):
    meta = read_json(Path(args.db) / 'database.json')
    return load_capture(args.data, meta['alignment'], meta['depth_scale'])


def aggregate(results):
    successful = [r for r in results if r['success']]
    summary = dict(total_queries=len(results), localized_queries=len(successful),
        pose_return_rate=len(successful)/max(len(results), 1),
        note='Errors are conditional on a returned pose. Threshold rates use ALL queries; failures count as misses.',
        validity='Internal held-out scan test with shared Polycam optimization. Not independent phone-photo accuracy.')
    for key in ['position_error_m', 'rotation_error_deg']:
        values = [r[key] for r in successful]
        summary[key] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
            p90=float(np.percentile(values, 90))) if values else None
    summary['accuracy_thresholds_all_queries'] = []
    for meters, degrees in [(.05, 5), (.25, 5), (.5, 10), (1., 20)]:
        hits = sum(r['success'] and r['position_error_m'] <= meters and r['rotation_error_deg'] <= degrees for r in results)
        summary['accuracy_thresholds_all_queries'].append(dict(position_m=meters, rotation_deg=degrees,
            hits=hits, total=len(results), rate=hits/max(len(results), 1)))
    for name in ['baseline_nearest_position_error_m', 'baseline_consensus_position_error_m']:
        values = [r[name] for r in results if name in r]
        summary[name] = dict(count=len(values), mean=float(np.mean(values)), median=float(np.median(values))) if values else None
    summary['median_localization_seconds'] = float(np.median([r['elapsed_seconds'] for r in results])) if results else None
    return summary


def evaluate(capture, db, out, s, exclude_neighbors=2, exclude_seconds=1., limit=None):
    if exclude_neighbors < 0 or exclude_seconds < 0:
        raise ValueError('Exclusion windows must be nonnegative')
    out = Path(out)
    (out / 'queries').mkdir(parents=True, exist_ok=True)
    ids = db.meta['query_ids'][:limit]
    if not ids:
        raise ValueError('No held-out queries')
    results = []
    for j, id in enumerate(ids):
        frame = capture.frames[id]
        if getattr(capture, 'dataset_format', '') == '7-scenes':
            # These are different official sequences, not adjacent timestamps.
            excluded = set(capture.query_ids)
        else:
            index = capture.ids.index(id)
            excluded = set(capture.ids[max(0, index-exclude_neighbors):index+exclude_neighbors+1])
            excluded.update(rid for rid in db.ids if abs(int(rid)-int(id))/1e6 <= exclude_seconds)
        r = localize(capture, db, rgb_image(frame.image), frame.K, s, excluded, sha256(frame.image))
        r.update(query_id=id, ground_truth=pose_dict(frame.T_world_cv))
        if r['success']:
            est = np.asarray(r['pose']['T_world_camera_cv'])
            pe, re = pose_errors(est, frame.T_world_cv)
            r.update(position_error_m=pe, rotation_error_deg=re)
            print(f"Query {j+1}/{len(ids)} {id}: {pe:.3f} m, {re:.2f} deg, {r['inliers']} inliers", flush=True)
        else:
            print(f"Query {j+1}/{len(ids)} {id}: FAILED ({r['reason']})", flush=True)
        if 'baseline_nearest' in r:
            p, a = pose_errors(np.asarray(r['baseline_nearest']['T_world_camera_cv']), frame.T_world_cv)
            r.update(baseline_nearest_position_error_m=p, baseline_nearest_rotation_error_deg=a,
                baseline_consensus_position_error_m=float(np.linalg.norm(
                    np.array(r['baseline_consensus_position_m'])-frame.T_world_cv[:3, 3])))
        write_json(out / 'queries' / f'{id}.json', r)
        results.append(r)
    summary = aggregate(results)
    seven = getattr(capture, 'dataset_format', '') == '7-scenes'
    if seven:
        summary['validity'] = 'Sampled 7-Scenes official test sequences against sampled train sequences. Supplied TSDF mesh may use all sequences; RGB intrinsics are approximate. Not a full benchmark.'
    summary['protocol'] = dict(query_ids=ids, reference_count=len(db.ids), exclude_neighbors=exclude_neighbors,
                               exclude_seconds=exclude_seconds, seed=s.seed,
                               vocabulary_trained_on='reference images only', query_depth_used=False,
                               shared_map_optimization=True)
    summary['protocol'].update(dataset_format=getattr(capture, 'dataset_format', 'polycam'),
        split=db.meta['split'], rgb_calibration=('approximate fx=fy=525 unless overridden at import' if seven else 'Polycam corrected intrinsics'),
        reference_depth_source=('mesh rendered into RGB grid' if seven else 'scaled Polycam sensor depth'))
    if seven:
        summary['protocol'].update(exclude_neighbors=None, exclude_seconds=None,
            train_sequences=capture.meta['train_sequences'], test_sequences=capture.meta['test_sequences'])
    write_json(out / 'summary.json', summary)
    write_json(out / 'results.json', results)
    fields = ['query_id', 'success', 'reason', 'position_error_m', 'rotation_error_deg',
        'inliers', 'median_reprojection_px', 'elapsed_seconds', 'excluded_reference_count',
        'baseline_nearest_position_error_m', 'baseline_nearest_rotation_error_deg', 'baseline_consensus_position_error_m']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    scene_html(capture, out / 'localization.html', results, db.ids)
    evaluation_plot(capture, results, out / 'overview.png')
    print(f"Pose return rate: {summary['localized_queries']}/{summary['total_queries']}", flush=True)
    print(f"Results: {out.resolve()}", flush=True)
    return summary


def calibrate(args):
    if args.cols < 2 or args.rows < 2 or args.square_m <= 0:
        raise ValueError('Checkerboard rows/cols must be >= 2 and square-m must be positive')
    files = sorted(Path(args.images).glob('*'))
    object_points, image_points, used = [], [], []
    grid = np.zeros((args.cols*args.rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)*args.square_m
    size = None
    for p in files:
        if p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        rgb = rgb_image(p, orient=True)
        current = (rgb.shape[1], rgb.shape[0])
        if size and current != size:
            raise ValueError('All calibration images must have the same oriented resolution')
        size = current
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (args.cols, args.rows))
        if found:
            object_points.append(grid.copy())
            image_points.append(corners)
            used.append(p.name)
    if len(used) < 8:
        raise ValueError(f'Only {len(used)} checkerboards found; need at least 8 varied views')
    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    write_json(args.output, dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=size[0], height=size[1], distortion=dist.ravel().tolist(), rms_reprojection_px=float(rms), images=used))
    print(f'Calibration: {args.output}; RMS reprojection error {rms:.3f} px')


def parser():
    p = argparse.ArgumentParser(description='Polycam / 7-Scenes visual localization (CPU features + PnP).')
    p.add_argument('--threads', type=int, default=2, help='OpenCV threads, default 2')
    sub = p.add_subparsers(dest='command', required=True)
    web=sub.add_parser('web',help='Open room upload, automatic indexing and photo localization website')
    web.add_argument('--output',default='outputs/web')
    web.add_argument('--port',type=int,default=8765)
    web.add_argument('--no-browser',action='store_true')
    web.add_argument('--room',choices=['polycam','chess'],help='Initially select a bundled room')
    app = sub.add_parser('app', help='Open the local photo-upload and 3D room interface')
    app.add_argument('--data', default='data/room')
    app.add_argument('--output', default='outputs/app')
    app.add_argument('--port', type=int, default=8765)
    app.add_argument('--no-browser', action='store_true')
    prep = sub.add_parser('prepare', help='Extract capture ZIP and copy GLB')
    prep.add_argument('--zip', required=True);prep.add_argument('--glb', required=True);prep.add_argument('--output', required=True)
    seven = sub.add_parser('prepare-7scenes', help='Import an official 7-Scenes scene folder or ZIP with separate train/test sequences')
    seven.add_argument('--source', required=True)
    seven.add_argument('--glb', required=True)
    seven.add_argument('--output', required=True)
    seven.add_argument('--train-stride', type=int, default=10)
    seven.add_argument('--test-stride', type=int, default=50)
    seven.add_argument('--rgb-focal', type=float, default=525., help='Approximate RGB focal length in pixels')
    seven.add_argument('--world-to-glb', help='JSON 4x4 rigid transform from original dataset world to the supplied GLB. Default diag(1,-1,-1,1) matches our Chess GLB.')
    def data_args(sp):
        sp.add_argument('--data', required=True, help='Prepared Polycam or 7-Scenes capture folder')
    def capture_args(sp):
        data_args(sp)
        sp.add_argument('--alignment', choices=['forward','inverse','identity'], default='forward')
        sp.add_argument('--depth-scale', type=float, default=.001)
    def solver_args(sp):
        sp.add_argument('--top-k', type=int, default=8)
        sp.add_argument('--ratio', type=float, default=.75)
        sp.add_argument('--reprojection-px', type=float, default=4.)
        sp.add_argument('--min-inliers', type=int, default=12)
        sp.add_argument('--seed', type=int, default=42)
    def build_args(sp):
        sp.add_argument('--query-every', type=int, default=5)
        sp.add_argument('--nfeatures', type=int, default=3000)
        sp.add_argument('--words', type=int, default=128)
    def evaluation_args(sp):
        sp.add_argument('--exclude-neighbors', type=int, default=2)
        sp.add_argument('--exclude-seconds', type=float, default=1.)
        sp.add_argument('--limit', type=int, help='Optional development query limit')
    ins = sub.add_parser('inspect', help='Check poses/depth against GLB; create offline camera viewer')
    capture_args(ins);ins.add_argument('--output', default='outputs/inspection')
    ins.add_argument('--skip-mesh-check', action='store_true')
    build = sub.add_parser('build', help='Create held-out split and reference-only retrieval database')
    capture_args(build);build_args(build);build.add_argument('--seed', type=int, default=42)
    build.add_argument('--output', default='outputs/database')
    ev = sub.add_parser('evaluate', help='Evaluate held-out query frames with temporal exclusion')
    data_args(ev);solver_args(ev);evaluation_args(ev)
    ev.add_argument('--db', required=True);ev.add_argument('--output', default='outputs/evaluation')
    loc = sub.add_parser('localize', help='Localize a new phone image; calibrated intrinsics required')
    data_args(loc);solver_args(loc)
    loc.add_argument('--db', required=True);loc.add_argument('--query', required=True)
    loc.add_argument('--intrinsics', required=True, help='JSON calibrated for oriented image resolution')
    loc.add_argument('--output', default='outputs/new_query')
    mobile = sub.add_parser('export-mobile', help='Export a reference-only .roomloc package for the offline iPhone app')
    data_args(mobile)
    mobile.add_argument('--db', required=True)
    mobile.add_argument('--output', required=True, help='Output .roomloc filename')
    mobile.add_argument('--name', default='Room')
    demo = sub.add_parser('demo', help='Inspect, build and evaluate end to end')
    capture_args(demo);build_args(demo);solver_args(demo);evaluation_args(demo)
    demo.add_argument('--output', default='outputs/demo')
    demo.add_argument('--skip-mesh-check', action='store_true')
    cal = sub.add_parser('calibrate', help='Calibrate camera from checkerboard photographs')
    cal.add_argument('--images', required=True);cal.add_argument('--cols', type=int, default=9)
    cal.add_argument('--rows', type=int, default=6);cal.add_argument('--square-m', type=float, default=.025)
    cal.add_argument('--output', default='camera_intrinsics.json')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cv2.setNumThreads(max(1, args.threads))
    try:
        if args.command == 'web':
            from .app import run_web
            run_web(args.output,args.port,not args.no_browser,args.room)
        elif args.command == 'app':
            from .app import run_app
            run_app(args.data, args.output, args.port, not args.no_browser)
        elif args.command == 'prepare':
            print(prepare_capture(args.zip, args.glb, args.output))
        elif args.command == 'prepare-7scenes':
            from .seven_scenes import prepare_seven_scenes
            G=read_json(args.world_to_glb) if args.world_to_glb else None
            print(prepare_seven_scenes(args.source,args.glb,args.output,args.train_stride,args.test_stride,G,args.rgb_focal))
        elif args.command == 'calibrate':
            calibrate(args)
        elif args.command == 'export-mobile':
            from .mobile import export_mobile
            capture = capture_for_database(args)
            print(export_mobile(capture, args.db, args.output, args.name))
        elif args.command == 'inspect':
            c = load_capture(args.data, args.alignment, args.depth_scale)
            report = audit(c, args.output, not args.skip_mesh_check)
            print(f"Frames: {report['frames']}; viewer: {Path(args.output)/'cameras.html'}")
            for w in report['warnings']:
                print('Note:', w)
        elif args.command == 'build':
            c = load_capture(args.data, args.alignment, args.depth_scale)
            m = build_database(c, args.output, args.query_every, args.nfeatures, args.words, args.seed)
            print(f"Built {len(m['reference_ids'])} references and {len(m['query_ids'])} held-out queries")
        elif args.command == 'evaluate':
            c = capture_for_database(args);db = Database(args.db, c)
            evaluate(c, db, args.output, settings(args), args.exclude_neighbors, args.exclude_seconds, args.limit)
        elif args.command == 'demo':
            c = load_capture(args.data, args.alignment, args.depth_scale)
            out = Path(args.output)
            print('Checking coordinate alignment and depth...', flush=True)
            report = audit(c, out / 'inspection', not args.skip_mesh_check)
            if report.get('alignment_matches_lowest_residual') is False:
                raise ValueError('Coordinate audit disagrees with selected alignment. Review inspection.json/cameras.html.')
            if not (out / 'database/database.json').is_file():
                build_database(c, out / 'database', args.query_every, args.nfeatures, args.words, args.seed)
            db = Database(out / 'database', c)
            expected = [args.query_every, args.nfeatures, args.words, args.seed]
            actual = [db.meta[k] for k in ['query_every', 'nfeatures', 'words', 'seed']]
            if expected != actual:
                raise ValueError('Existing demo database uses different settings. Use a fresh output directory.')
            evaluate(c, db, out / 'evaluation', settings(args), args.exclude_neighbors, args.exclude_seconds, args.limit)
        elif args.command == 'localize':
            c = capture_for_database(args);db = Database(args.db, c)
            rgb = rgb_image(args.query, orient=True)
            K, dist = query_calibration(args.intrinsics, (rgb.shape[1], rgb.shape[0]))
            if np.any(dist):
                rgb = cv2.undistort(rgb, K, dist, None, K)
            r = localize(c, db, rgb, K, settings(args), query_hash=sha256(args.query))
            r.update(query_id=Path(args.query).name, query_intrinsics=K.tolist(), input_distortion=dist.tolist())
            out = Path(args.output)
            write_json(out / 'pose.json', r)
            scene_html(c, out / 'localization.html', [r], db.ids)
            overlay = Image.fromarray(rgb)
            draw = ImageDraw.Draw(overlay)
            for x, y in r.get('inlier_query_pixels', []):
                draw.ellipse((x-4,y-4,x+4,y+4), outline='#22c55e', width=2)
            overlay.save(out / 'inliers.jpg')
            if r['success']:
                print('Position (m):', r['pose']['position_m'])
                print('Quaternion (x,y,z,w):', r['pose']['quaternion_xyzw'])
                print(f"Inliers: {r['inliers']}; median reprojection: {r['median_reprojection_px']:.2f} px")
            else:
                print('Localization failed:', r['reason'])
                sys.exit(2)
    except (ValueError, KeyError, OSError, cv2.error) as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

