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

from .features import build_database, Database
from .localization import localize, Settings, pose_dict, pose_errors
from .photo import read_photo, intrinsics_for_photo, refine_focal, validate_calibration
from .polycam import read_json, write_json, sha256
from .capture import load_capture
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
            raise ValueError('缺少 room.glb，请先导入房间模型。')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path).resolve() if db_path else self.output / 'database'
        self.config_path = self.output / 'camera.json'
        self.calibration = validate_calibration(read_json(self.config_path)) if self.config_path.exists() else None
        self.lock = threading.RLock()
        self.state = dict(phase='starting', message='正在读取房间模型…', progress=0, ready=False)
        self.jobs = {}
        self.scene = None
        self.db = None
        self.learned = None
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
            mesh.update(opacity=.90, name='房间模型')
            ids = getattr(self.capture, 'reference_ids', self.capture.ids)
            centers = np.array([self.capture.frames[id].T_world_cv[:3, 3] for id in ids])
            figure = go.Figure([mesh, go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                mode='markers', name='扫描相机', visible=False, text=ids,
                marker=dict(size=3, color='#6b8d88'), hoverinfo='text')])
            vertices = np.column_stack([mesh.x, mesh.y, mesh.z])
            figure.update_layout(meta=dict(bounds=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]))
            self.scene = figure.to_json().encode()
            if not (self.db_path/'database.json').is_file():
                self._state(phase='building', message='首次使用：正在建立房间参考库…', progress=0)
                def progress(done, total):
                    self._state(progress=done/total, message=f'正在建立参考库：{done}/{total} 张照片')
                build_database(self.capture, self.db_path, query_every=None, progress=progress)
            self._state(phase='loading', message='正在加载定位参考库…', progress=.98)
            self.db = Database(self.db_path, self.capture)
            self._state(phase='ready', message='房间已就绪，可以放入照片', progress=1, ready=True)
        except Exception as e:
            self._state(phase='error', message=f'初始化失败：{e}', ready=False)

    def status(self):
        with self.lock:
            return dict(self.state, room_frames=len(self.capture.ids),
                references=len(self.db.ids) if self.db else 0, has_calibration=self.calibration is not None,
                dataset_format=getattr(self.capture, 'dataset_format', 'polycam'),
                scene_name=getattr(self.capture, 'meta', {}).get('scene', '你的房间'),
                test_frames=len(getattr(self.capture, 'query_ids', [])),
                calibration_size=([self.calibration['width'], self.calibration['height']] if self.calibration else None))

    def save_calibration(self, value):
        c = validate_calibration(value)
        with self.lock:
            write_json(self.config_path, c)
            self.calibration = c
        return {'saved': True, 'width': c['width'], 'height': c['height']}

    def submit(self, image_bytes=None, filename='photo', mode='auto', sample=False):
        with self.lock:
            if not self.state['ready']:
                raise ValueError('房间参考库还没准备好，请稍等。')
            if sum(j['status'] in ['queued', 'running'] for j in self.jobs.values()) >= 2:
                raise ValueError('已有定位任务正在处理，请等待完成。')
            id = uuid.uuid4().hex
            folder = self.output/'queries'/id
            folder.mkdir(parents=True, exist_ok=True)
            if not sample:
                if not image_bytes:
                    raise ValueError('请先选择一张照片。')
                (folder/'source.image').write_bytes(image_bytes)
            self.jobs[id] = dict(id=id, status='queued', message='等待定位…', progress=0,
                filename=Path(filename).name[:160], sample=sample)
            calibration = dict(self.calibration) if self.calibration else None
            self.worker.submit(self._process, id, mode, sample, calibration)
            return dict(self.jobs[id])

    def job(self, id):
        with self.lock:
            if id not in self.jobs:
                raise KeyError('找不到这次定位任务。')
            return dict(self.jobs[id])

    def _update_job(self, id, **kwargs):
        with self.lock:
            self.jobs[id].update(kwargs)

    def _process(self, id, mode, sample, calibration):
        folder = self.output/'queries'/id
        started = time.perf_counter()
        try:
            self._update_job(id, status='running', progress=.1, message='正在读取照片…')
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
                              label='7-Scenes RGB 近似参数' if seven else '示例照片的已知相机参数', estimated=seven,
                              note='来自官方测试序列，未加入参考库；RGB 焦距采用数据集常用近似值。' if heldout else '示例来自参考照片，仅用于演示；匹配时会排除这张原图。')
                query_hash = sha256(frame.image)
                # Interactive sample: reject the exact frame by hash, while using
                # neighboring references. Temporal exclusions belong to evaluation.
                self._update_job(id, filename=f'{frame_id}.color.png' if seven else f'{frame_id}.jpg')
            else:
                rgb, metadata = read_photo(folder/'source.image')
                if mode == 'dataset':
                    if not seven:raise ValueError('当前房间不是 7-Scenes，请选择自动估算或导入标定参数。')
                    if (metadata['width'],metadata['height'])!=(640,480):
                        raise ValueError('7-Scenes 参数只适用于未裁剪的 640×480 数据集照片。手机照片请选择自动估算或导入标定。')
                    candidates,distortion,_=intrinsics_for_photo(metadata,'saved',self.capture.profile)
                    camera=dict(source='7-scenes',label='7-Scenes RGB 近似参数',estimated=True,
                        note='RGB 焦距采用数据集常用近似值；不是实测相机标定。')
                else:
                    candidates, distortion, camera = intrinsics_for_photo(metadata, mode, calibration)
                query_hash = sha256(folder/'source.image')
            if seven:
                excluded.update(self.capture.query_ids)
            if np.any(distortion):
                rgb = cv2.undistort(rgb, candidates[0], distortion, None, candidates[0])
            Image.fromarray(rgb).save(folder/'query.jpg', quality=92)
            self._update_job(id, progress=.3, message='正在匹配房间特征并计算拍照位置…',
                preview_url=f'/results/{id}/query.jpg')
            result = localize(self.capture, self.db, rgb, candidates[0], Settings(), excluded,
                              query_hash, candidate_intrinsics=candidates)
            result['method'] = 'RootSIFT + visual-word retrieval + depth PnP'
            if not result['success'] and result.get('query_features',0) >= 24:
                original_failure = {k:result.get(k) for k in ['method','reason','query_features','correspondences','inliers','retrieval']}
                try:
                    def enhanced_progress(message, fraction):
                        self._update_job(id, progress=fraction, message=message)
                    if self.learned is None:
                        from .learned import LearnedLocalizer
                        self.learned = LearnedLocalizer(self.capture,self.db.ids,self.output/'learned_features',enhanced_progress)
                    else:
                        self.learned.progress = enhanced_progress
                    result = self.learned.localize(rgb,candidates,excluded,query_hash,self.db.meta['reference_image_hashes'])
                    result['initial_attempt'] = original_failure
                except Exception as e:
                    result['enhancement_error'] = str(e)
                    result['enhancement_help'] = '增强匹配不可用。请重新运行启动脚本安装依赖，并在首次使用时保持联网。'
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
            result.update(query_id=self.jobs[id]['filename'], photo_metadata=metadata, camera=camera, sample=sample)
            result['urls'] = dict(preview=f'/results/{id}/query.jpg', pose=f'/results/{id}/pose.json')
            if result['success']:
                self._update_job(id, progress=.85, message='正在把拍照位置标到房间模型里…')
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
                message='已找到拍照位置' if result['success'] else '这张照片暂时无法可靠定位', result=result)
        except Exception as e:
            self._update_job(id, status='error', progress=1, message=f'处理失败：{e}')

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
            raise ValueError('上传内容为空或超过 40 MB。请先把照片导出为 JPG。')
        body=self.rfile.read(length)
        if len(body)!=length:raise ValueError('上传中断，请重试。')
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
                self.respond(202, app.submit(sample=True, filename='示例照片'))
            elif self.path == '/api/localize':
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data;'):
                    raise ValueError('请选择照片后再定位。')
                message = BytesParser(policy=email_policy).parsebytes(
                    f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode()+body)
                file, mode, filename = None, 'auto', 'photo'
                if not message.is_multipart():raise ValueError('上传格式不正确。')
                for part in message.iter_parts():
                    name = part.get_param('name', header='content-disposition')
                    if name == 'image':file=part.get_payload(decode=True);filename=part.get_filename() or filename
                    elif name == 'mode':mode=part.get_payload(decode=True).decode()
                self.respond(202, app.submit(file, filename, mode))
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
        raise OSError('端口不可用，请关闭旧程序或使用 --port 选择其他端口。')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'\n房间照片定位已启动：{url}\n照片仅在本机处理。关闭程序请在终端按 Control+C。\n', flush=True)
    app.start()
    if open_browser:
        threading.Timer(.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n正在关闭…', flush=True)
    finally:
        server.server_close()
        app.close()


def run_web(output='outputs/web',port=8765,open_browser=True,initial=None):
    from .rooms import RoomHub
    cv2.setNumThreads(2)
    project=Path(__file__).resolve().parent.parent
    builtins={
        'polycam':dict(name='我的房间 · Polycam',data=project/'data/room',output=project/'outputs/app'),
        'chess':dict(name='Chess · 7-Scenes',data=project/'data/seven_scenes/chess',output=project/'outputs/7scenes_chess')}
    app=RoomHub(output,builtins,initial)
    server=None
    for candidate in range(port,port+10):
        try:server=LocalServer(('127.0.0.1',candidate),app);break
        except OSError:continue
    if server is None:app.close();raise OSError('端口不可用，请关闭旧程序或使用 --port 选择端口。')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'\n房间上传与定位网页：{url}\n在页面添加 GLB 和参考 ZIP。退出请按 Control+C。\n',flush=True)
    app.start()
    if open_browser:threading.Timer(.8,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.close()
