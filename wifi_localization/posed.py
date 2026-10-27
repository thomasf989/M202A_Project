"""Reference photos with explicit intrinsics and poses, lifted on the GLB."""
from pathlib import Path
import hashlib
import re
import numpy as np
from PIL import Image
from .polycam import Frame, rigid_matrix, camera_matrix, read_json, sha256
from .seven_scenes import SevenScenesCapture


class PosedCapture:
    dataset_format='posed-images'
    split_description='Only frames marked reference/train enter the map; explicit test frames remain held out'
    world_points=SevenScenesCapture.world_points

    def __init__(self, root):
        self.root=Path(root).resolve();path=self.root/'references.json';self.meta=read_json(path)
        if self.meta.get('format')!='posed-images-v1':raise ValueError('references.json: format must be posed-images-v1')
        self.A=rigid_matrix(self.meta.get('world_to_glb',np.eye(4)),'world_to_glb');self.raw_alignment=self.A
        self.alignment='dataset';self.depth_scale=1.;self.glb=self.root/'room.glb'
        if not self.glb.is_file():raise ValueError('Missing room.glb')
        self.frames={};self.reference_ids=[];self.query_ids=[]
        convention=self.meta.get('camera_convention','opencv')
        if convention not in ['opencv','opengl']:raise ValueError('camera_convention must be opencv or opengl')
        C=np.diag([1.,-1.,-1.,1.]) if convention=='opengl' else np.eye(4)
        for i,r in enumerate(self.meta['frames']):
            id=str(r.get('id',f'photo-{i:06d}'))
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',id) or id in self.frames:raise ValueError('Invalid or duplicate photo ID')
            image=(self.root/r['image']).resolve()
            if not image.is_relative_to(self.root) or not image.is_file():raise ValueError(f'Missing or unsafe reference image: {id}')
            c=r.get('intrinsics',self.meta.get('intrinsics'))
            if c is None:raise ValueError(f'Missing camera intrinsics: {id}')
            K=camera_matrix(c);w,h=int(c['width']),int(c['height'])
            if min(w,h)<2:raise ValueError('Invalid image size')
            if np.any(c.get('distortion',[0])):raise ValueError('Reference photos must be undistorted before import')
            with Image.open(image) as im:
                if im.size!=(w,h):raise ValueError(f'Photo size differs from intrinsics: {id}')
                if im.getexif().get(274,1)!=1:raise ValueError(f'Reference image must be stored upright, with pose/intrinsics for those pixels: {id}')
            T=rigid_matrix(r['camera_to_world'],id)
            self.frames[id]=Frame(id,image,path,path,path,K,self.A@T@C,w,h)
            split=r.get('split','reference')
            if split not in ['reference','train','test']:raise ValueError('Frame split must be reference, train or test')
            (self.query_ids if split=='test' else self.reference_ids).append(id)
        self.ids=list(self.frames)
        if len(self.reference_ids)<4:raise ValueError('Need at least four posed reference photos')
        self._mesh=None;self._depth_cache={}

    def fingerprint(self):
        h=hashlib.sha256((self.root/'references.json').read_bytes());h.update(sha256(self.glb).encode());h.update(b'posed-mesh-depth-v1')
        for id in self.ids:h.update(sha256(self.frames[id].image).encode())
        return h.hexdigest()
