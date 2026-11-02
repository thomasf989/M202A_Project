"""Local browser application. No room data or photos leave this computer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
import webbrowser

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plotly.offline import get_plotlyjs
import plotly.graph_objects as go
import trimesh

from .features import build_database, Database, extract
from .localization import localize, Settings, pose_dict, pose_errors
from .photo import read_photo, intrinsics_for_photo, refine_focal, validate_calibration
from .polycam import read_json, write_json, sha256
from .capture import load_capture
from .multiset import read_multiset_pose, compare_to_multiset
from .visualization import mesh_trace, scene_html

ASSETS = Path(__file__).with_name('web')
MAX_UPLOAD = 40 * 1024 * 1024


def export_marked_glb(capture, result, path):
    """Preserve the textured room and add a red camera center/frustum in GLB coordinates."""
    scene = trimesh.load_scene(capture.glb)
    for geometry in scene.geometry.values():
        material = getattr(geometry.visual, 'material', None)
        if isinstance(material, trimesh.visual.material.PBRMaterial):
            # Room scans are diffuse surfaces, rather than unlit metal. Some
            # viewers otherwise render an unspecified metallic factor as white.
            material.metallicFactor = 0.
            material.roughnessFactor = .9
            material.doubleSided = True
    T = np.asarray(result['pose']['T_world_camera_cv'])
    color = [239, 68, 68, 255]
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=.045)
    sphere.visual.face_colors = color
    scene.add_geometry(sphere, geom_name='localized_camera_center', transform=T)
    corners = np.array([[-.13,-.10,.35],[.13,-.10,.35],[.13,.10,.35],[-.13,.10,.35]])
    lines = [(np.zeros(3), c) for c in corners] + list(zip(corners, np.roll(corners, -1, axis=0)))
    for i, (a, b) in enumerate(lines):
        segment = trimesh.creation.cylinder(radius=.007, segment=[a, b], sections=8)
        segment.visual.face_colors = color
        scene.add_geometry(segment, geom_name=f'localized_camera_frustum_{i}', transform=T)
    scene.metadata['localized_camera'] = result['pose']
    scene.export(path, file_type='glb')


class LocalApp:
    def __init__(self, data, output, db_path=None):
        self.capture = load_capture(data)
        if not self.capture.glb.is_file():
            raise ValueError('room.glb is missing. Import a room model first.')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path).resolve() if db_path else self.output / 'database'
        self.config_path = self.output / 'camera.json'
        self.calibration = validate_calibration(read_json(self.config_path)) if self.config_path.exists() else None
        self.lock = threading.RLock()
        self.state = dict(phase='starting', message='Reading the room model…', progress=0, ready=False)
        self.jobs = {}
        self.scene = None
        self.db = None
        self.learned = None
        self.retriever = None
        self.pipeline = None
        self.sample_index = 0
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='room-localizer')
        self.initialization = None
        self.plotly_js = get_plotlyjs().encode()

    def start(self):
        self.initialization = self.worker.submit(self._initialize)

    def _state(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def _initialize(self):
        try:
            mesh = mesh_trace(self.capture.glb)
            mesh.update(opacity=.90, name='Room model')
            ids = getattr(self.capture, 'reference_ids', self.capture.ids)
            centers = np.array([self.capture.frames[id].T_world_cv[:3, 3] for id in ids])
            figure = go.Figure([mesh, go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                mode='markers', name='Scan cameras', visible=False, text=ids,
                marker=dict(size=3, color='#6b8d88'), hoverinfo='text')])
            vertices = np.column_stack([mesh.x, mesh.y, mesh.z])
            figure.update_layout(meta=dict(bounds=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]))
            self.scene = figure.to_json().encode()
            def build_reference_database(message='First use: building the room reference library…'):
                self._state(phase='building', message=message, progress=0)
                if self.db_path.exists():
                    shutil.rmtree(self.db_path)
                def progress(done, total):
                    self._state(progress=done/total, message=f'Building reference library: {done}/{total} photos')
                build_database(self.capture, self.db_path, query_every=None, progress=progress)
            if not (self.db_path/'database.json').is_file():
                build_reference_database()
            self._state(phase='loading', message='Loading the localization reference library…', progress=.98)
            try:
                self.db = Database(self.db_path, self.capture)
            except ValueError as e:
                # Algorithm/map-depth upgrades intentionally change the capture
                # fingerprint. Rebuild stale feature databases automatically
                # instead of leaving an existing uploaded room unusable.
                if 'fingerprint' in str(e).lower() or 'changed' in str(e).lower() or 'Rebuild' in str(e):
                    build_reference_database('A localization algorithm or map-depth upgrade was detected. Rebuilding the reference library automatically…')
                    self.db = Database(self.db_path, self.capture)
                else:
                    raise
            from .pipeline import LocalizationPipeline
            self.pipeline = LocalizationPipeline(self.capture, self.db, self.output/'robust_features')
            self._state(phase='ready', message='The room is ready. You can add a photo.', progress=1, ready=True)
        except Exception as e:
            self._state(phase='error', message=f'Initialization failed: {e}', ready=False)

    def status(self):
        with self.lock:
            return dict(self.state, room_frames=len(self.capture.ids),
                references=len(self.db.ids) if self.db else 0, has_calibration=self.calibration is not None,
                dataset_format=getattr(self.capture, 'dataset_format', 'polycam'),
                scene_name=getattr(self.capture, 'meta', {}).get('scene', 'Your room'),
                test_frames=len(getattr(self.capture, 'query_ids', [])),
                calibration_size=([self.calibration['width'], self.calibration['height']] if self.calibration else None))

    def save_calibration(self, value):
        c = validate_calibration(value)
        with self.lock:
            write_json(self.config_path, c)
            self.calibration = c
        return {'saved': True, 'width': c['width'], 'height': c['height']}

    def submit(self, image_bytes=None, filename='photo', mode='auto', sample=False, multiset_pose_bytes=None, solver_mode='meshloc'):
        with self.lock:
            if not self.state['ready']:
                raise ValueError('The room reference library is not ready yet. Please wait.')
            if sum(j['status'] in ['queued', 'running'] for j in self.jobs.values()) >= 2:
                raise ValueError('A localization job is already running. Please wait for it to finish.')
            id = uuid.uuid4().hex
            folder = self.output/'queries'/id
            folder.mkdir(parents=True, exist_ok=True)
            if not sample:
                if not image_bytes:
                    raise ValueError('Select a photo first.')
                (folder/'source.image').write_bytes(image_bytes)
            if multiset_pose_bytes:
                if len(multiset_pose_bytes) > 1024*1024:
                    raise ValueError('MultiSet pose JSON is unexpectedly large.')
                try:
                    baseline=json.loads(multiset_pose_bytes.decode('utf-8'))
                    read_multiset_pose(baseline)
                except Exception as e:
                    raise ValueError(f'MultiSet pose JSON is invalid: {e}') from e
                write_json(folder/'multiset_pose.json',baseline)
            solver_mode=str(solver_mode or 'meshloc').lower()
            if solver_mode not in {'meshloc','meshloc-paper','auto','legacy'}:
                raise ValueError('Unknown localization algorithm mode.')
            self.jobs[id] = dict(id=id, status='queued', message='Waiting to localize…', progress=0,
                filename=Path(filename).name[:160], sample=sample, solver_mode=solver_mode)
            calibration = dict(self.calibration) if self.calibration else None
            self.worker.submit(self._process, id, mode, sample, calibration, solver_mode)
            return dict(self.jobs[id])

    def job(self, id):
        with self.lock:
            if id not in self.jobs:
                raise KeyError('This localization job could not be found.')
            return dict(self.jobs[id])

    def _update_job(self, id, **kwargs):
        with self.lock:
            self.jobs[id].update(kwargs)

    def _process(self, id, mode, sample, calibration, solver_mode):
        folder = self.output/'queries'/id
        started = time.perf_counter()
        try:
            self._update_job(id, status='running', progress=.1, message='Reading the photo…')
            excluded = set()
            seven = getattr(self.capture, 'dataset_format', '') == '7-scenes'
            benchmark_frame = None
            if sample:
                heldout=seven and bool(self.capture.query_ids)
                if heldout:
                    frame_id=self.capture.query_ids[self.sample_index % len(self.capture.query_ids)]
                    self.sample_index+=1
                    benchmark_frame=self.capture.frames[frame_id]
                else:
                    frame_id = self.capture.ids[min(17, len(self.capture.ids)-1)]
                frame = self.capture.frames[frame_id]
                rgb, metadata = read_photo(frame.image)
                # The selected sample is already <=1280 px and is undistorted.
                candidates, distortion = [frame.K], np.zeros(5)
                camera = dict(source='7-scenes' if seven else 'sample',
                              label='Approximate 7-Scenes RGB parameters' if seven else 'Known camera parameters for the sample photo', estimated=seven,
                              note='From the official test sequence and not included in the reference library; the RGB focal length uses a common dataset approximation.' if heldout else 'The sample comes from the reference photos and is for demonstration only; the original image is excluded during matching.')
                query_hash = sha256(frame.image)
                # Interactive sample: reject the exact frame by hash, while using
                # neighboring references. Temporal exclusions belong to evaluation.
                self._update_job(id, filename=f'{frame_id}.color.png' if seven else f'{frame_id}.jpg')
            else:
                rgb, metadata = read_photo(folder/'source.image')
                if mode == 'dataset':
                    if not seven:raise ValueError('The current room is not a 7-Scenes room. Select automatic estimation or import calibration parameters.')
                    if (metadata['width'],metadata['height'])!=(640,480):
                        raise ValueError('7-Scenes parameters apply only to uncropped 640×480 dataset photos. For phone photos, select automatic estimation or import calibration.')
                    candidates,distortion,_=intrinsics_for_photo(metadata,'saved',self.capture.profile)
                    camera=dict(source='7-scenes',label='Approximate 7-Scenes RGB parameters',estimated=True,
                        note='The RGB focal length uses a common dataset approximation; it is not a measured camera calibration.')
                else:
                    candidates, distortion, camera = intrinsics_for_photo(metadata, mode, calibration)
                query_hash = sha256(folder/'source.image')
            if seven:
                excluded.update(self.capture.query_ids)
            if np.any(distortion):
                rgb = cv2.undistort(rgb, candidates[0], distortion, None, candidates[0])
            Image.fromarray(rgb).save(folder/'query.jpg', quality=92)
            self._update_job(id, progress=.3, message='Matching room features and calculating the camera position…',
                preview_url=f'/results/{id}/query.jpg')
            # Robust hierarchy: multi-crop DINOv2 -> pose-neighbour expansion
            # -> ALIKED/LightGlue + F-RANSAC -> metric PnP, with a hybrid SIFT
            # fallback on the same shortlist. MultiSet is never used inside
            # this solver; it remains only an optional external baseline.
            def enhanced_progress(message, fraction):
                self._update_job(id, progress=fraction, message=message)
            if self.pipeline is None:
                from .pipeline import LocalizationPipeline
                self.pipeline = LocalizationPipeline(self.capture, self.db, self.output/'robust_features', enhanced_progress)
            else:
                self.pipeline.progress = enhanced_progress
            robust_settings = Settings(top_k=20, ratio=.78, reprojection_px=5.,
                                       min_inliers=10, min_inlier_ratio=.08,
                                       iterations=6000, seed=42)
            result = self.pipeline.localize(rgb, candidates, excluded, query_hash,
                                            settings=robust_settings, enhanced=True,
                                            solver_mode=solver_mode)
            if not result.get('success'):
                result['enhancement_help'] = (
                    'No reliable pose yet. Check reference coverage, use the same physical room/model, '
                    'and prefer photos containing textured walls/furniture rather than a blank wall.')
            if camera['estimated'] and camera['source'] != '7-scenes':
                refine_focal(result, (rgb.shape[1], rgb.shape[0]))
            if seven:
                # Identify known test photos ONLY AFTER localization, for
                # evaluation display. No query pose is given to either solver.
                if benchmark_frame is None:
                    benchmark_frame=next((self.capture.frames[qid] for qid in self.capture.query_ids
                        if sha256(self.capture.frames[qid].image)==query_hash),None)
                result['dataset_format']='7-scenes'
                result['depth_source']='reference mesh rendered into RGB grid'
                if benchmark_frame is not None:
                    result['ground_truth']=pose_dict(benchmark_frame.T_world_cv)
                    result['evaluation_note']='Sampled official test sequence; approximate RGB intrinsics; supplied mesh may use all sequences.'
                    if result['success']:
                        pe,re=pose_errors(np.asarray(result['pose']['T_world_camera_cv']),benchmark_frame.T_world_cv)
                        result.update(position_error_m=pe,rotation_error_deg=re)
            multiset_path=folder/'multiset_pose.json'
            if multiset_path.is_file():
                try:
                    result.update(compare_to_multiset(result,read_json(multiset_path)))
                except Exception as e:
                    result['multiset_error']=str(e)
            result.update(query_id=self.jobs[id]['filename'], photo_metadata=metadata, camera=camera, sample=sample, requested_solver_mode=solver_mode)
            result['urls'] = dict(preview=f'/results/{id}/query.jpg', pose=f'/results/{id}/pose.json')
            if result['success']:
                self._update_job(id, progress=.85, message='Marking the camera position in the room model…')
                export_marked_glb(self.capture, result, folder/'localized_room.glb')
                scene_html(self.capture, folder/'localization.html', [result], self.db.ids)
                overlay = Image.fromarray(rgb)
                draw = ImageDraw.Draw(overlay)
                for x, y in result.get('inlier_query_pixels', []):
                    draw.ellipse((x-3,y-3,x+3,y+3), outline='#22c55e', width=2)
                overlay.save(folder/'inliers.jpg', quality=90)
                result['urls'].update(model=f'/results/{id}/localized_room.glb',
                    viewer=f'/results/{id}/localization.html', inliers=f'/results/{id}/inliers.jpg')
            result['total_seconds'] = time.perf_counter()-started
            write_json(folder/'pose.json', result)
            self._update_job(id, status='done', progress=1,
                message='Camera position found' if result['success'] else 'This photo could not be localized reliably', result=result)
        except Exception as e:
            self._update_job(id, status='error', progress=1, message=f'Processing failed: {e}')

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)

    def result_path(self,id,name):
        return self.output/'queries'/id/name


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Avoid logging file contents or browser tokens; leave terminal usable.
        if args and str(args[0]).startswith('POST'):
            print(fmt % args)

    def permitted(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:
            self.respond(403, {'error': 'Only local connections are supported.'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{h}' for h in allowed}:
            self.respond(403, {'error': 'Cross-origin requests are not supported.'})
            return False
        return True

    def respond(self, code, obj, content_type='application/json; charset=utf-8'):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode() if isinstance(obj, (dict, list)) else obj
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_path(self, path, mime=None):
        if not path.is_file():
            self.respond(404, {'error': 'File not found'})
            return
        self.respond(200, path.read_bytes(), mime or mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

    def do_GET(self):
        if not self.permitted():return
        path = self.path.split('?', 1)[0]
        app = self.server.app
        if path == '/':self.send_path(ASSETS/'index.html', 'text/html; charset=utf-8')
        elif path in ['/app.js', '/style.css', '/reference-format.json']:
            self.send_path(ASSETS/path[1:])
        elif path == '/plotly.js':self.respond(200, app.plotly_js, 'text/javascript; charset=utf-8')
        elif path == '/api/status':self.respond(200, app.status())
        elif path == '/api/rooms' and hasattr(app,'list_rooms'):self.respond(200,app.list_rooms())
        elif re.fullmatch(r'/api/rooms/[a-z0-9_-]{1,64}',path) and hasattr(app,'room_status'):
            try:self.respond(200,app.room_status(path.rsplit('/',1)[1]))
            except KeyError:self.respond(404,{'error':'Unknown room'})
        elif re.fullmatch(r'/api/uploads/[0-9a-f]{32}',path) and hasattr(app,'upload_status'):
            try:self.respond(200,app.upload_status(path.rsplit('/',1)[1]))
            except ValueError as e:self.respond(404,{'error':str(e)})
        elif path == '/api/scene':
            self.respond(200, app.scene, 'application/json') if app.scene else self.respond(503, {'error':'Room is loading'})
        elif re.fullmatch(r'/api/jobs/[0-9a-f]{32}', path):
            try:self.respond(200, app.job(path.rsplit('/', 1)[1]))
            except KeyError:self.respond(404, {'error':'Unknown job'})
        elif re.fullmatch(r'/results/[0-9a-f]{32}/(query.jpg|pose.json|localized_room.glb|localization.html|inliers.jpg)', path):
            try:self.send_path(app.result_path(path.split('/')[2],path.split('/')[3]))
            except KeyError:self.respond(404,{'error':'Unknown result'})
        else:self.respond(404, {'error':'Not found'})

    def read_body(self, maximum=MAX_UPLOAD):
        try:length = int(self.headers.get('Content-Length', 0))
        except ValueError:raise ValueError('Invalid content length')
        if not 0 < length <= maximum:
            raise ValueError('The upload is empty or exceeds 40 MB. Export the photo as a JPG first.')
        body=self.rfile.read(length)
        if len(body)!=length:raise ValueError('The upload was interrupted. Please try again.')
        return body

    def do_PUT(self):
        if not self.permitted():return
        if not re.fullmatch(r'/api/uploads/[0-9a-f]{32}',self.path) or not hasattr(self.server.app,'append_upload'):
            self.respond(404,{'error':'Not found'});return
        try:
            from .rooms import CHUNK_BYTES
            offset=int(self.headers.get('Upload-Offset','-1'))
            self.respond(200,self.server.app.append_upload(self.path.rsplit('/',1)[1],offset,self.read_body(CHUNK_BYTES)))
        except (ValueError,OSError) as e:self.respond(400,{'error':str(e)})

    def do_POST(self):
        if not self.permitted():return
        try:
            app = self.server.app
            body = self.read_body()
            if self.path == '/api/uploads' and hasattr(app,'create_upload'):
                self.respond(201,app.create_upload(json.loads(body)))
            elif self.path == '/api/rooms' and hasattr(app,'create_room'):
                self.respond(202,app.create_room(json.loads(body)))
            elif self.path == '/api/rooms/select' and hasattr(app,'select_room'):
                self.respond(200,app.select_room(json.loads(body)['id']))
            elif self.path == '/api/calibration':
                self.respond(200, app.save_calibration(json.loads(body)))
            elif self.path == '/api/sample':
                self.respond(202, app.submit(sample=True, filename='Sample photo', solver_mode='meshloc'))
            elif self.path == '/api/localize':
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data;'):
                    raise ValueError('Select a photo before starting localization.')
                message = BytesParser(policy=email_policy).parsebytes(
                    f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode()+body)
                file, mode, filename, multiset_pose, solver_mode = None, 'auto', 'photo', None, 'meshloc'
                if not message.is_multipart():raise ValueError('The upload format is invalid.')
                for part in message.iter_parts():
                    name = part.get_param('name', header='content-disposition')
                    if name == 'image':file=part.get_payload(decode=True);filename=part.get_filename() or filename
                    elif name == 'mode':mode=part.get_payload(decode=True).decode()
                    elif name == 'multiset_pose':multiset_pose=part.get_payload(decode=True)
                    elif name == 'solver_mode':solver_mode=part.get_payload(decode=True).decode()
                self.respond(202, app.submit(file, filename, mode, multiset_pose_bytes=multiset_pose, solver_mode=solver_mode))
            else:self.respond(404, {'error':'Not found'})
        except (ValueError, OSError, KeyError) as e:
            self.respond(400, {'error': str(e)})


def run_app(data='data/room', output='outputs/app', port=8765, open_browser=True):
    cv2.setNumThreads(2)
    app = LocalApp(data, output)
    server = None
    for candidate in range(port, port+10):
        try:
            server = LocalServer(('127.0.0.1', candidate), app)
            break
        except OSError:
            continue
    if server is None:
        app.close()
        raise OSError('The port is unavailable. Close the previous program or use --port to select another port.')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'\nRoom photo localization started: {url}\nPhotos are processed locally only. Press Control+C in the terminal to stop the program.\n', flush=True)
    app.start()
    if open_browser:
        threading.Timer(.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down…', flush=True)
    finally:
        server.server_close()
        app.close()


def run_web(output='outputs/web',port=8765,open_browser=True,initial=None):
    from .rooms import RoomHub
    cv2.setNumThreads(2)
    project=Path(__file__).resolve().parent.parent
    builtins={
        'polycam':dict(name='My Room · Polycam',data=project/'data/room',output=project/'outputs/app'),
        'chess':dict(name='Chess · 7-Scenes',data=project/'data/seven_scenes/chess',output=project/'outputs/7scenes_chess')}
    app=RoomHub(output,builtins,initial)
    server=None
    for candidate in range(port,port+10):
        try:server=LocalServer(('127.0.0.1',candidate),app);break
        except OSError:continue
    if server is None:app.close();raise OSError('The port is unavailable. Close the previous program or use --port to select a port.')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'\nRoom upload and localization page: {url}\nAdd a GLB and reference ZIP on the page. Press Control+C to exit.\n',flush=True)
    app.start()
    if open_browser:threading.Timer(.8,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.close()
