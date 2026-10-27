"""7-Scenes adapter. Official sequence splits; RGB-aligned mesh depth.

Original Kinect RGB and depth are NOT calibrated/registered. We therefore
render reference depth from the supplied aligned mesh at the RGB intrinsics
instead of indexing the raw depth PNG at RGB pixel coordinates. The commonly
used RGB focal length 525 is an approximation, not a measured calibration.
"""
from pathlib import Path
import hashlib
import io
import re
import shutil
import zipfile
import tempfile
import numpy as np
from PIL import Image
from .polycam import Frame, rigid_matrix, camera_matrix, read_json, write_json, sha256

DEFAULT_WORLD_TO_GLB = np.diag([1., -1., -1., 1.])


def dataset_pose(T, label):
    """Project small KinectFusion numerical drift onto SO(3).

    Chess originals have orthogonality residuals up to 0.00030. Reject
    reflections and substantial scale/shear; preserve original files.
    """
    T=np.asarray(T,dtype=float).copy()
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-5):
        raise ValueError(f'{label}: invalid camera pose')
    R=T[:3,:3]
    if np.max(np.abs(R.T@R-np.eye(3)))>.001 or np.linalg.det(R)<=0:
        raise ValueError(f'{label}: camera rotation has excessive scale/shear or reflection')
    U,_,Vt=np.linalg.svd(R);T[:3,:3]=U@Vt
    return rigid_matrix(T,label)


def _split(text):
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if not rows or any(not re.fullmatch(r'sequence\d+', r) for r in rows):
        raise ValueError('Invalid 7-Scenes split file')
    return [f'seq-{int(r[8:]):02d}' for r in rows]


def prepare_seven_scenes(source, glb, output, train_stride=10, test_stride=50,
                         world_to_glb=None, rgb_focal=525., require_test=True):
    """Read an extracted official scene, outer scene ZIP or our preview bundle.

    A subset may contain only some official sequences. At least four train
    frames and one test frame are required. Train/test membership always
    comes from the official split files, never every-nth-frame sampling.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if train_stride < 1 or test_stride < 1 or not np.isfinite(rgb_focal) or rgb_focal <= 0:
        raise ValueError('Strides and RGB focal length must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output must be empty: {output}')
    if not Path(glb).is_file():
        raise ValueError('Missing corresponding GLB')
    G = rigid_matrix(DEFAULT_WORLD_TO_GLB if world_to_glb is None else world_to_glb, 'world_to_glb')
    output.mkdir(parents=True, exist_ok=True)
    records, skipped, seen = [], [], set()
    try:
        if source.is_dir():
            split_paths = list(source.glob('TrainSplit.txt')) or list(source.glob('*/TrainSplit.txt'))
            if len(split_paths) != 1:
                raise ValueError('Select one scene folder containing TrainSplit.txt and TestSplit.txt')
            root = split_paths[0].parent
            train = _split((root/'TrainSplit.txt').read_text());test = _split((root/'TestSplit.txt').read_text())
            context = None
            def read(name):return (root/name).read_bytes()
            names = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
        else:
            context = zipfile.ZipFile(source)
            names0 = context.namelist()
            roots = [n[:-len('TrainSplit.txt')] for n in names0 if n.endswith('TrainSplit.txt') and '__MACOSX' not in n]
            if len(roots) != 1:
                raise ValueError('ZIP must contain one 7-Scenes scene and its official split files')
            prefix = roots[0]
            def read(name):return context.read(prefix+name)
            names = [n[len(prefix):] for n in names0 if n.startswith(prefix)]
            train = _split(read('TrainSplit.txt').decode());test = _split(read('TestSplit.txt').decode())
        try:
            if set(train)&set(test):raise ValueError('Train and test sequences overlap')
            def copy_frame(seq, stem, reader):
                id = f'{seq}_{stem}'
                if id in seen:raise ValueError(f'Duplicate frame {id}')
                seen.add(id)
                T = np.loadtxt(io.BytesIO(reader(stem+'.pose.txt')))
                if not np.isfinite(T).all():
                    skipped.append(id);return
                dataset_pose(T, id)
                rgb = reader(stem+'.color.png');depth = reader(stem+'.depth.png')
                with Image.open(io.BytesIO(rgb)) as im:
                    if im.size != (640, 480):raise ValueError(f'Expected 640x480 RGB: {id}')
                with Image.open(io.BytesIO(depth)) as im:
                    if im.size != (640, 480):raise ValueError(f'Expected 640x480 depth: {id}')
                for suffix, data in [('color.png',rgb),('depth.png',depth),('pose.txt',reader(stem+'.pose.txt'))]:
                    dest=output/seq/f'{stem}.{suffix}';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                records.append(dict(id=id, sequence=seq, stem=stem, split='train' if seq in train else 'test'))
            for seq in train+test:
                stride = train_stride if seq in train else test_stride
                nested = f'{seq}.zip'
                if nested in names:
                    # Official outer archive stores per-sequence ZIPs. Handle one
                    # at a time; never extract untrusted archive paths.
                    with tempfile.TemporaryFile() as temp:
                        with (context.open(prefix+nested) if context is not None else (root/nested).open('rb')) as src:
                            shutil.copyfileobj(src,temp,1024*1024)
                        temp.seek(0)
                        with zipfile.ZipFile(temp) as zz:
                            if len(zz.infolist())>100000 or sum(i.file_size for i in zz.infolist())>12*1024**3:
                                raise ValueError('7-Scenes nested sequence exceeds size limits')
                            stems=sorted({Path(n).name[:-len('.color.png')] for n in zz.namelist() if n.endswith('.color.png')})
                            for stem in stems[::stride]:copy_frame(seq,stem,lambda n:zz.read(f'{seq}/{n}'))
                else:
                    pattern=re.compile(rf'^{re.escape(seq)}[/_](frame-\d+)\.color\.png$')
                    found={m.group(1):n[:-len('.color.png')] for n in names if (m:=pattern.fullmatch(n))}
                    for stem in sorted(found)[::stride]:
                        base=found[stem]
                        copy_frame(seq,stem,lambda n,base=base,stem=stem:read(base+n[len(stem):]))
            if sum(r['split']=='train' for r in records)<4 or (require_test and not any(r['split']=='test' for r in records)):
                raise ValueError('Need at least 4 training frames and 1 official test frame. The six-photo preview bundle has no test sequence; use the full scene or the included prepared Chess data.')
            shutil.copy2(glb,output/'room.glb')
            for name, seqs in [('TrainSplit.txt',train),('TestSplit.txt',test)]:
                (output/name).write_text('\n'.join('sequence'+str(int(s[4:])) for s in seqs)+'\n')
            profile=dict(width=640,height=480,fx=float(rgb_focal),fy=float(rgb_focal),cx=320.,cy=240.,distortion=[0,0,0,0,0])
            write_json(output/'camera_intrinsics.json',profile)
            manifest=dict(format='7-scenes-v1',scene=Path(glb).stem,world_to_glb=G.tolist(),
                rgb_intrinsics=profile,depth_source='mesh_rendered_rgb',rgb_calibration='approximate',
                train_sequences=train,test_sequences=test,train_stride=train_stride,test_stride=test_stride,
                frames=records,skipped_invalid_pose=skipped,localization_only=not any(r['split']=='test' for r in records),
                note='Sampled official sequence split; supplied mesh may use all sequences. This is not a full benchmark.')
            write_json(output/'seven_scenes.json',manifest)
        finally:
            if context is not None:context.close()
    except Exception:
        # No manifest is written on failure; leave files available for inspection.
        raise
    return output


class SevenScenesCapture:
    dataset_format = '7-scenes'

    def __init__(self, root):
        self.root=Path(root).resolve();self.meta=read_json(self.root/'seven_scenes.json')
        if self.meta.get('format')!='7-scenes-v1' or self.meta.get('depth_source')!='mesh_rendered_rgb':
            raise ValueError('Unsupported 7-Scenes manifest')
        self.A=rigid_matrix(self.meta['world_to_glb'],'world_to_glb');self.raw_alignment=self.A
        self.alignment='dataset';self.depth_scale=.001;self.glb=self.root/'room.glb'
        if not self.glb.is_file():raise ValueError('Missing 7-Scenes room.glb')
        self.profile=self.meta['rgb_intrinsics'];K=camera_matrix(self.profile)
        self.frames={};self.reference_ids=[];self.query_ids=[]
        for r in self.meta['frames']:
            seq,stem=r['sequence'],r['stem']
            if not re.fullmatch(r'seq-\d+',seq) or not re.fullmatch(r'frame-\d+',stem) or r['id']!=f'{seq}_{stem}':
                raise ValueError('Unsafe or inconsistent 7-Scenes frame ID')
            if r['id'] in self.frames:raise ValueError('Duplicate 7-Scenes frame ID')
            group=self.meta['train_sequences'] if r['split']=='train' else self.meta['test_sequences']
            if r['split'] not in ['train','test'] or seq not in group or set(self.meta['train_sequences'])&set(self.meta['test_sequences']):
                raise ValueError('Invalid or overlapping official sequence split')
            base=self.root/seq/stem
            image=Path(str(base)+'.color.png');dep=Path(str(base)+'.depth.png');pose=Path(str(base)+'.pose.txt')
            for p in [image,dep,pose]:
                if not p.is_file():raise ValueError(f'Missing 7-Scenes frame input: {p}')
            with Image.open(image) as im:
                if im.size!=(self.profile['width'],self.profile['height']):raise ValueError('RGB/intrinsic size mismatch')
            T=dataset_pose(np.loadtxt(pose),str(pose))
            self.frames[r['id']]=Frame(r['id'],image,pose,dep,dep,K.copy(),self.A@T,int(self.profile['width']),int(self.profile['height']))
            (self.reference_ids if r['split']=='train' else self.query_ids).append(r['id'])
        self.ids=sorted(self.frames)
        if len(self.reference_ids)<4 or (not self.query_ids and not self.meta.get('localization_only')):
            raise ValueError('7-Scenes requires separate reference and test frames')
        self._mesh=None;self._depth_cache={}

    def fingerprint(self):
        h=hashlib.sha256((self.root/'seven_scenes.json').read_bytes());h.update(sha256(self.glb).encode())
        h.update(b'mesh-rgb-depth-v1')
        for id in self.ids:
            for p in [self.frames[id].image,self.frames[id].camera,self.frames[id].depth]:h.update(sha256(p).encode())
        return h.hexdigest()

    def world_points(self, frame, pixels, min_confidence=127, edge_threshold=.15):
        from .raster import mesh_depth
        if self._mesh is None:
            from .visualization import load_mesh
            self._mesh=load_mesh(self.glb)
        if frame.id not in self._depth_cache:
            if len(self._depth_cache)>=4:self._depth_cache.pop(next(iter(self._depth_cache)))
            self._depth_cache[frame.id]=mesh_depth(self._mesh,frame.T_world_cv,frame.K,frame.width,frame.height)
        d=self._depth_cache[frame.id];uv=np.asarray(pixels,dtype=float).reshape(-1,2)
        ij=np.rint(uv).astype(int);x,y=np.clip(ij,[1,1],[frame.width-2,frame.height-2]).T
        valid=(ij[:,0]>=1)&(ij[:,0]<frame.width-1)&(ij[:,1]>=1)&(ij[:,1]<frame.height-1)
        patches=np.stack([d[y+dy,x+dx] for dy in [-1,0,1] for dx in [-1,0,1]],axis=1)
        z=d[y,x];valid &= np.isfinite(patches).all(1)&(z>.15)&(z<5.)
        with np.errstate(invalid='ignore'):
            valid &= np.ptp(patches,axis=1)<edge_threshold
        safe_z=np.where(valid,z,0)
        pc=np.column_stack([(uv[:,0]-frame.K[0,2])/frame.K[0,0]*safe_z,(uv[:,1]-frame.K[1,2])/frame.K[1,1]*safe_z,safe_z])
        return pc@frame.T_world_cv[:3,:3].T+frame.T_world_cv[:3,3],valid
