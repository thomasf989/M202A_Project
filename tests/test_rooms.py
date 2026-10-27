"""Room upload/index/localize/export across HTTP and process restarts."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import zipfile
import numpy as np
from wifi_localization.rooms import RoomHub,CHUNK_BYTES,safe_archive,prepare_room,validate_glb
import struct
from wifi_localization.capture import load_capture
from wifi_localization.app import LocalServer

PROJECT=Path(__file__).resolve().parents[1]


class UploadContracts(unittest.TestCase):
    def test_offsets_limits_restart_and_invalid_paths(self):
        with tempfile.TemporaryDirectory() as d:
            hub=RoomHub(d)
            self.assertEqual(hub.status()['phase'],'empty')
            with self.assertRaises(ValueError):hub.create_upload(dict(kind='model',filename='bad.glb',size=513*1024*1024))
            with self.assertRaises(ValueError):hub.create_upload(dict(kind='references',filename='data.jpg',size=10))
            up=hub.create_upload(dict(kind='model',filename='test.glb',size=10))
            hub.append_upload(up['id'],0,b'abcd')
            with self.assertRaises(ValueError):hub.append_upload(up['id'],0,b'wrong')
            with self.assertRaises(ValueError):hub.upload_status('../../file')
            hub.close();hub=RoomHub(d)
            try:
                self.assertEqual(hub.upload_status(up['id'])['offset'],4)
                final=hub.append_upload(up['id'],4,b'efghij');self.assertTrue(final['complete'])
                with self.assertRaises(ValueError):hub.append_upload(up['id'],10,b'overflow')
            finally:hub.close()

    def test_unsafe_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('../outside.txt','bad')
            with self.assertRaises(ValueError):safe_archive(path)
            self.assertFalse((Path(d).parent/'outside.txt').exists())

    def test_glb_requires_embedded_resources(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'external.glb'
            doc=json.dumps({'asset':{'version':'2.0'},'buffers':[{'uri':'../../outside.bin'}]}).encode()
            doc+=b' '*((-len(doc))%4)
            p.write_bytes(struct.pack('<4sII',b'glTF',2,20+len(doc))+struct.pack('<II',len(doc),0x4e4f534a)+doc)
            with self.assertRaisesRegex(ValueError,'外部文件'):validate_glb(p)
        model=PROJECT/'data/seven_scenes/chess/room.glb'
        if model.exists():validate_glb(model)

    def test_interrupted_import_does_not_prevent_startup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);id='a'*32;(root/'rooms'/id/'data').mkdir(parents=True)
            (root/'rooms.json').write_text(json.dumps(dict(active=id,rooms={id:dict(id=id,name='Interrupted',phase='importing')})))
            hub=RoomHub(root)
            try:
                hub.start();self.assertFalse(hub.status()['ready']);self.assertEqual(hub.status()['phase'],'error')
            finally:hub.close()


class RoomHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=PROJECT/'data/seven_scenes/chess';cls.db=PROJECT/'outputs/7scenes_chess'
        if not (cls.data/'seven_scenes.json').is_file():raise unittest.SkipTest('Bundled Chess capture is required')
        cls.capture=load_capture(cls.data);cls.temp=tempfile.TemporaryDirectory()
        cls.config={'chess':dict(name='Chess',data=cls.data,output=cls.db)}
        cls.hub=RoomHub(cls.temp.name,cls.config);cls.hub.start()
        cls.hub.current().initialization.result(timeout=120)
        cls.server=LocalServer(('127.0.0.1',0),cls.hub)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.hub.close();cls.temp.cleanup()

    def request(self,path,value=None,method=None,headers=None):
        body=json.dumps(value).encode() if isinstance(value,dict) else value
        with urlopen(Request(self.base+path,data=body,method=method,headers={'Content-Type':'application/json',**(headers or {})}),timeout=60) as r:
            data=r.read();return json.loads(data) if 'application/json' in r.headers.get('Content-Type','') else data

    def upload(self,path,kind):
        p=Path(path);meta=self.request('/api/uploads',dict(kind=kind,filename=p.name,size=p.stat().st_size))
        with p.open('rb') as f:
            first=f.read(1000);self.request('/api/uploads/'+meta['id'],first,'PUT',{'Upload-Offset':'0'})
            resumed=self.request('/api/uploads/'+meta['id']);self.assertEqual(resumed['offset'],len(first))
            with self.assertRaises(HTTPError):self.request('/api/uploads/'+meta['id'],b'bad','PUT',{'Upload-Offset':'0'})
            offset=len(first)
            while chunk:=f.read(CHUNK_BYTES):
                row=self.request('/api/uploads/'+meta['id'],chunk,'PUT',{'Upload-Offset':str(offset)})
                offset=row['offset']
        self.assertTrue(row['complete']);return meta['id']

    def wait_room(self,id):
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            row=self.request('/api/rooms/'+id)
            if row.get('ready') or row['phase']=='error' or row.get('reuse_id'):return row
            time.sleep(.05)
        self.fail('Room preparation timed out')

    def test_upload_posed_photos_build_localize_switch_and_restart(self):
        folder=Path(self.temp.name);archive=folder/'references.zip';model=folder/'room.glb'
        model.write_bytes(self.capture.glb.read_bytes());frames=[]
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for id in self.capture.reference_ids[:6]:
                f=self.capture.frames[id];name=f'images/{id}.png';z.writestr(name,f.image.read_bytes())
                frames.append(dict(id=id,image=name,camera_to_world=f.T_world_cv.tolist()))
            z.writestr('references.json',json.dumps(dict(format='posed-images-v1',intrinsics=self.capture.profile,frames=frames)))
        mid=self.upload(model,'model');rid=self.upload(archive,'references')
        room=self.request('/api/rooms',dict(name='Uploaded Chess photos',model_upload=mid,references_upload=rid,options={'format':'auto'}))
        row=self.wait_room(room['id']);self.assertTrue(row['ready'],row)
        self.assertEqual(row['references'],6);self.assertEqual(row['dataset_format'],'posed-images')
        app=self.hub.current();target=self.capture.frames[self.capture.query_ids[0]]
        boundary='new-room-photo-test'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nsaved\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n').encode()+target.image.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        self.request('/api/calibration',self.capture.profile)
        job=self.request('/api/localize',body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
        app.worker.submit(lambda:None).result(timeout=120)
        result=self.request('/api/jobs/'+job['id']);self.assertEqual(result['status'],'done',result)
        r=result['result'];self.assertTrue(r['success'],r.get('reason'))
        self.assertIn('model',r['urls'])
        before=self.request(r['urls']['model']);self.assertEqual(before[:4],b'glTF')
        # A result continues to refer to its own room after the selected room changes.
        self.request('/api/rooms/select',{'id':'chess'})
        self.assertEqual(self.request(r['urls']['model']),before)
        self.assertFalse(self.request('/api/status')['has_calibration'])
        self.request('/api/rooms/select',{'id':room['id']})
        self.assertTrue(self.request('/api/status')['has_calibration'])
        # Same source files and options reuse the map rather than rebuilding it.
        same=self.request('/api/rooms',dict(name='Same files',model_upload=mid,references_upload=rid,options={'format':'auto'}))
        duplicate=self.wait_room(same['id']);self.assertEqual(duplicate['reuse_id'],room['id'])
        # The persisted registry and reference database reopen in a new manager.
        reopened=RoomHub(folder,self.config)
        try:
            reopened.start();reopened.current().initialization.result(timeout=120)
            self.assertTrue(reopened.status()['ready']);self.assertEqual(reopened.status()['references'],6)
        finally:reopened.close()

    def test_reference_schema_and_origin_boundary(self):
        schema=self.request('/reference-format.json');self.assertEqual(schema['format'],'posed-images-v1')
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/uploads',dict(kind='model',filename='x.glb',size=10),headers={'Origin':'https://unrelated.example'})
        self.assertEqual(caught.exception.code,403)

    def test_prepared_7scenes_archive_preserves_test_split(self):
        folder=Path(self.temp.name);archive=folder/'prepared_chess.zip'
        meta=json.loads((self.data/'seven_scenes.json').read_text())
        keep=set(self.capture.reference_ids[:4]+self.capture.query_ids[:1])
        meta['frames']=[r for r in meta['frames'] if r['id'] in keep]
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('chess/seven_scenes.json',json.dumps(meta))
            for name in ['TrainSplit.txt','TestSplit.txt']:z.writestr('chess/'+name,(self.data/name).read_bytes())
            for id in keep:
                f=self.capture.frames[id]
                for p in [f.image,f.camera,f.depth]:z.writestr('chess/'+p.relative_to(self.data).as_posix(),p.read_bytes())
        dest=folder/'prepared_import'
        self.assertEqual(prepare_room(archive,self.capture.glb,dest,{'format':'auto'}),'7-scenes')
        capture=load_capture(dest)
        self.assertEqual(len(capture.reference_ids),4);self.assertEqual(len(capture.query_ids),1)
        self.assertFalse(set(capture.reference_ids)&set(capture.query_ids))

    def test_z_missing_camera_information_is_reported_without_false_ready(self):
        archive=Path(self.temp.name)/'ordinary_photos.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('photo.png',self.capture.frames[self.capture.reference_ids[0]].image.read_bytes())
        mid=self.upload(self.capture.glb,'model');rid=self.upload(archive,'references')
        room=self.request('/api/rooms',dict(name='Incomplete',model_upload=mid,references_upload=rid))
        result=self.wait_room(room['id']);self.assertEqual(result['phase'],'error')
        self.assertFalse(result['ready']);self.assertIn('相机参数和位姿',result['message'])
        self.request('/api/rooms/select',{'id':'chess'});self.assertTrue(self.request('/api/status')['ready'])


if __name__=='__main__':unittest.main()
