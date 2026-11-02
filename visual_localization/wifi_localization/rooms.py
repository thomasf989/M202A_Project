"""Persistent room registry and bounded, resumable browser uploads."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import struct
import threading
import time
import uuid
import zipfile
import numpy as np
from plotly.offline import get_plotlyjs
from .polycam import prepare_capture, read_json, write_json, sha256
from .capture import load_capture

CHUNK_BYTES=4*1024*1024
MAX_MODEL_BYTES=512*1024*1024
MAX_ARCHIVE_BYTES=4*1024**3
MAX_EXPANDED_BYTES=12*1024**3
RESULT_FILES={'query.jpg','pose.json','localized_room.glb','localization.html','inliers.jpg','room.roomloc'}


def safe_archive(archive):
    """Reject traversal/symlinks and excessive expansion before extraction."""
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist())>100000:raise ValueError('ZIP 文件条目过多。')
        if sum(i.file_size for i in z.infolist())>MAX_EXPANDED_BYTES:raise ValueError('ZIP 解压后超过 12 GB。')
        names=[]
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if '\\' in info.filename or p.is_absolute() or '..' in p.parts or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('ZIP 包含不安全的文件路径。')
            if not info.is_dir() and not any(x.startswith('.') or x=='__MACOSX' for x in p.parts):names.append(info.filename)
        return names


def validate_glb(path):
    with Path(path).open('rb') as f:
        header=f.read(12)
        chunk_header=f.read(8)
        if len(chunk_header)!=8:raise ValueError('GLB 文件不完整。')
        json_length,json_type=struct.unpack('<II',chunk_header)
        if json_type!=0x4e4f534a or json_length>64*1024*1024:raise ValueError('GLB 头信息无效。')
        document=json.loads(f.read(json_length))
        for resource in document.get('buffers',[])+document.get('images',[]):
            uri=resource.get('uri')
            if uri is not None and (not isinstance(uri,str) or not uri.startswith('data:')):
                raise ValueError('GLB 必须内嵌几何和纹理，不能引用外部文件。')
    if len(header)!=12:raise ValueError('GLB 文件不完整。')
    magic,version,length=struct.unpack('<4sII',header)
    if magic!=b'glTF' or version!=2 or length!=Path(path).stat().st_size:raise ValueError('请选择有效的 GLB 2.0 模型。')
    import trimesh
    scene=trimesh.load_scene(path,file_type='glb')
    if not scene.geometry:raise ValueError('模型中没有可用的三维网格。')
    for mesh in scene.geometry.values():
        if not hasattr(mesh,'faces') or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError('模型包含无效网格。')


def prepare_room(archive, model, output, options):
    names=safe_archive(archive);validate_glb(model);output=Path(output)
    roots={}
    for kind,marker in [('polycam','optimized_roomplan.json'),('7-scenes','TrainSplit.txt'),('posed-images','references.json')]:
        matches=[n[:-len(marker)] for n in names if PurePosixPath(n).name==marker]
        if matches:roots[kind]=matches
    requested=options.get('format','auto')
    if requested=='auto':
        if len(roots)!=1:raise ValueError('无法唯一识别数据格式。请上传完整 Polycam 原始数据、7-Scenes 数据，或照片与 references.json。只有普通照片还缺少相机参数和位姿。')
        kind=next(iter(roots))
    else:kind=requested
    if kind not in roots or len(roots[kind])!=1:raise ValueError('ZIP 应只包含一个房间及其相机信息。')
    prefix=roots[kind][0]
    if kind=='polycam':
        prepare_capture(archive,model,output)
    elif kind=='7-scenes':
        from .seven_scenes import prepare_seven_scenes
        transform=options.get('world_to_glb')
        stride=int(options.get('train_stride',10));test_stride=int(options.get('test_stride',50))
        if not 1<=stride<=10000 or not 1<=test_stride<=10000:raise ValueError('抽样间隔需在 1 到 10000 之间。')
        if prefix+'seven_scenes.json' in names:
            with zipfile.ZipFile(archive) as z:
                for name in names:
                    if name.startswith(prefix):
                        rel=PurePosixPath(name[len(prefix):]);dest=output/str(rel);dest.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(name) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
            shutil.copy2(model,output/'room.glb')
        else:
            prepare_seven_scenes(archive,model,output,stride,test_stride,transform,float(options.get('rgb_focal',525)),require_test=False)
    elif kind=='posed-images':
        with zipfile.ZipFile(archive) as z:
            for name in names:
                if name.startswith(prefix):
                    rel=PurePosixPath(name[len(prefix):])
                    dest=output/str(rel);dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        shutil.copy2(model,output/'room.glb')
    else:raise ValueError('不支持的数据格式。')
    capture=load_capture(output)
    if len(getattr(capture,'reference_ids',capture.ids))<4:raise ValueError('至少需要 4 张有效参考照片。')
    if kind=='polycam':
        from scipy.spatial import cKDTree
        from .visualization import load_mesh
        tree=cKDTree(load_mesh(capture.glb).vertices);points=[]
        for id in capture.ids[::max(1,len(capture.ids)//8)]:
            frame=capture.frames[id];yy,xx=np.mgrid[frame.height//6:frame.height:frame.height//5,frame.width//6:frame.width:frame.width//5]
            xyz,valid=capture.world_points(frame,np.column_stack([xx.ravel(),yy.ravel()]));points.extend(xyz[valid])
        if len(points)<8 or float(np.median(tree.query(np.asarray(points))[0]))>.20:
            raise ValueError('GLB 与参考照片的三维坐标不一致。请使用同次扫描导出的原始数据和模型，保留坐标及米制尺度。')
    return kind


class RoomHub:
    def __init__(self, storage, builtins=None, initial=None):
        self.root=Path(storage).resolve();self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'uploads').mkdir(exist_ok=True);(self.root/'rooms').mkdir(exist_ok=True)
        self.lock=threading.RLock();self.upload_lock=threading.RLock()
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='room-import')
        self.apps={};self.owners={};self.futures=[];self.builtins=builtins or {}
        self.plotly_js=get_plotlyjs().encode()
        index=self.root/'rooms.json';stored=read_json(index) if index.exists() else {}
        self.entries=stored.get('rooms',{});self.active=stored.get('active')
        for id,config in self.builtins.items():
            if Path(config['data']).is_dir():
                self.entries[id]=dict(id=id,name=config['name'],builtin=True,phase='saved',message='选择后加载',created=0)
        for id,e in self.entries.items():
            if e.get('builtin'):continue
            if e.get('phase') in ['importing','building']:
                e.update(phase='saved' if self._data(id).exists() else 'error',message='可重新选择以继续准备房间')
        if initial in self.entries:self.active=initial
        if self.active not in self.entries:self.active=next(iter(self.entries),None)
        self._save()

    def _save(self):
        with self.lock:
            temp=self.root/'rooms.json.tmp';write_json(temp,dict(active=self.active,rooms=self.entries));temp.replace(self.root/'rooms.json')

    def _data(self,id):return self.root/'rooms'/id/'data'

    def _paths(self,id):
        e=self.entries[id]
        if e.get('builtin'):
            config=self.builtins[id];return Path(config['data']),Path(config['output'])
        return self._data(id),self.root/'rooms'/id/'output'

    def start(self):
        if self.active:
            try:self.select_room(self.active)
            except ValueError:pass

    def _open(self,id):
        from .app import LocalApp
        if id not in self.apps:
            data,output=self._paths(id);app=LocalApp(data,output);self.apps[id]=app;app.start()
        return self.apps[id]

    def current(self):
        with self.lock:
            app=self.apps.get(self.active)
            if app is None:raise ValueError('请先添加或选择已准备好的房间。')
            return app

    @property
    def scene(self):
        with self.lock:
            app=self.apps.get(self.active);return app.scene if app else None

    def room_status(self,id):
        with self.lock:
            if id not in self.entries:raise KeyError('找不到这个房间。')
            e=dict(self.entries[id]);app=self.apps.get(id)
            if app:
                e.update(app.status());e['room_id']=id
            e.setdefault('ready',False);e.setdefault('progress',0.)
            return e

    def status(self):
        with self.lock:
            if not self.active:return dict(phase='empty',ready=False,progress=0,message='先上传模型与参考数据，建立你的第一个房间',room_id=None,room_frames=0,references=0,has_calibration=False,dataset_format=None,scene_name='尚未选择房间')
            e=self.room_status(self.active)
            e.update(room_id=self.active,scene_name=self.entries[self.active]['name'])
            e.setdefault('has_calibration',False);e.setdefault('references',0);e.setdefault('room_frames',0)
            return e

    def list_rooms(self):
        with self.lock:
            return dict(active=self.active,rooms=[self.room_status(id) for id,e in self.entries.items() if not e.get('reuse_id')])

    def select_room(self,id):
        with self.lock:
            if id not in self.entries:raise ValueError('找不到这个房间。')
            entry=self.entries[id]
            if entry.get('reuse_id'):return self.select_room(entry['reuse_id'])
            if entry.get('phase')=='importing':self.active=id;self._save();return self.status()
            if entry.get('phase')=='error':raise ValueError(entry.get('message','请重新上传房间数据。'))
            try:self._open(id)
            except Exception as e:
                entry.update(phase='error',message=f'房间准备未完成，请重新上传：{e}');self._save()
                raise ValueError(entry['message']) from e
            self.active=id;self._save();return self.status()

    def create_upload(self,value):
        kind=value.get('kind');name=Path(str(value.get('filename',''))).name[:160];size=value.get('size')
        maximum=MAX_MODEL_BYTES if kind=='model' else MAX_ARCHIVE_BYTES
        if kind not in ['model','references'] or type(size) is not int or not 0<size<=maximum:raise ValueError('请选择 GLB（最大 512 MB）或参考 ZIP（最大 4 GB）。')
        if not name.lower().endswith('.glb' if kind=='model' else '.zip'):raise ValueError('文件类型不正确。')
        with self.upload_lock:
            id=uuid.uuid4().hex;meta=dict(id=id,kind=kind,filename=name,size=size,created=time.time())
            (self.root/'uploads'/f'{id}.part').touch();write_json(self.root/'uploads'/f'{id}.json',meta)
            return dict(meta,offset=0,complete=False,chunk_bytes=CHUNK_BYTES)

    def upload_status(self,id):
        if not isinstance(id,str) or len(id)!=32 or any(c not in '0123456789abcdef' for c in id):raise ValueError('无效上传编号。')
        with self.upload_lock:
            try:meta=read_json(self.root/'uploads'/f'{id}.json')
            except FileNotFoundError:raise ValueError('上传记录不存在，请重新选择文件。')
            offset=(self.root/'uploads'/f'{id}.part').stat().st_size
            return dict(meta,offset=offset,complete=offset==meta['size'],chunk_bytes=CHUNK_BYTES)

    def append_upload(self,id,offset,body):
        with self.upload_lock:
            row=self.upload_status(id)
            if offset!=row['offset']:raise ValueError('上传偏移不一致，请重试以续传。')
            if not 0<len(body)<=CHUNK_BYTES or offset+len(body)>row['size']:raise ValueError('上传分块大小不正确。')
            with (self.root/'uploads'/f'{id}.part').open('ab') as f:f.write(body)
            return self.upload_status(id)

    def create_room(self,value):
        name=str(value.get('name','')).strip()[:80] or '新房间'
        model=self.upload_status(value.get('model_upload'));refs=self.upload_status(value.get('references_upload'))
        if model['kind']!='model' or refs['kind']!='references' or not model['complete'] or not refs['complete']:raise ValueError('请先完成模型和参考数据上传。')
        options=value.get('options',{})
        if not isinstance(options,dict):raise ValueError('导入选项格式不正确。')
        options={k:options[k] for k in ['format','train_stride','test_stride','rgb_focal','world_to_glb'] if k in options}
        with self.lock:
            if any(not f.done() for f in self.futures):raise ValueError('已有房间正在建库，请等待完成。')
            id=uuid.uuid4().hex
            self.entries[id]=dict(id=id,name=name,builtin=False,phase='importing',message='正在检查文件与相机信息…',progress=.02,created=time.time())
            self.active=id;self._save()
            self.futures.append(self.worker.submit(self._import,id,model['id'],refs['id'],options))
            return self.room_status(id)

    def _import(self,id,model_id,refs_id,options):
        try:
            model=self.root/'uploads'/f'{model_id}.part';refs=self.root/'uploads'/f'{refs_id}.part'
            key=hashlib.sha256((sha256(model)+sha256(refs)+json.dumps(options,sort_keys=True)).encode()).hexdigest()
            with self.lock:
                same=next((eid for eid,e in self.entries.items() if eid!=id and e.get('source_key')==key and (self._paths(eid)[1]/'database/database.json').is_file()),None)
                if same:
                    self.entries[id].update(reuse_id=same,phase='saved',message='相同数据已有参考库，直接复用。')
                    self.select_room(same);return
            kind=prepare_room(refs,model,self._data(id),options)
            # Reuse a validated, matching built-in map when available. Never
            # load uploaded feature files or pickle-based caches.
            capture=load_capture(self._data(id));fingerprint=capture.fingerprint()
            expected=getattr(capture,'reference_ids',capture.ids)
            for config in self.builtins.values():
                source=Path(config['output']);meta_path=source/'database/database.json'
                if not meta_path.exists():continue
                meta=read_json(meta_path)
                if meta.get('capture_fingerprint')==fingerprint and meta.get('reference_ids')==expected:
                    dest=self._paths(id)[1];dest.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(source/'database',dest/'database')
                    if (source/'learned_features').is_dir():shutil.copytree(source/'learned_features',dest/'learned_features')
                    break
            with self.lock:
                self.entries[id].update(phase='building',message='文件检查完成，正在建立参考库…',format=kind,source_key=key)
                app=self._open(id);self._save()
            app.initialization.result()
            with self.lock:
                state=app.status();self.entries[id].update(phase='ready' if state['ready'] else 'error',message=state['message']);self._save()
        except Exception as e:
            with self.lock:
                self.entries[id].update(phase='error',message=f'房间准备失败：{e}');self._save()

    def submit(self,*args,**kwargs):
        app=self.current();job=app.submit(*args,**kwargs)
        with self.lock:self.owners[job['id']]=app
        return job

    def submit_mobile(self):
        with self.lock:
            app=self.current()
            name=self.entries[self.active]['name']
            job=app.submit_mobile(name)
            self.owners[job['id']]=app
        return job

    def job(self,id):
        with self.lock:app=self.owners.get(id)
        if app is None:raise KeyError('找不到这个任务。')
        return app.job(id)

    def result_path(self,id,name):
        if name not in RESULT_FILES:raise KeyError('未知结果文件。')
        with self.lock:
            if id in self.owners:return self.owners[id].output/'queries'/id/name
            for room_id in self.entries:
                path=self._paths(room_id)[1]/'queries'/id/name
                if path.is_file():return path
        raise KeyError('结果文件不存在。')

    def save_calibration(self,value):return self.current().save_calibration(value)

    def close(self):
        self.worker.shutdown(wait=True,cancel_futures=True)
        for app in self.apps.values():app.close()

