"""Camera handling and local HTTP upload/export integration contracts."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
from wifi_localization.photo import read_photo, intrinsics_for_photo, refine_focal
from wifi_localization.localization import pose_dict


class PhotoTests(unittest.TestCase):
    def test_saved_calibration_resize_and_wrong_aspect(self):
        metadata=dict(width=2048,height=1536,processing_width=1024,processing_height=768)
        calibration=dict(width=2048,height=1536,fx=1400,fy=1400,cx=1023.5,cy=767.5)
        K, dist, info=intrinsics_for_photo(metadata,'saved',calibration)
        np.testing.assert_allclose(K[0],[[700,0,511.5],[0,700,383.5],[0,0,1]])
        self.assertFalse(info['estimated'])
        with self.assertRaises(ValueError):
            intrinsics_for_photo(dict(metadata,width=1536,height=2048),'saved',calibration)

    def test_exif_orientation_and_automatic_estimate_label(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'oriented.jpg';exif=Image.Exif();exif[274]=6;exif[41989]=24
            Image.new('RGB',(1200,900),'white').save(p,exif=exif)
            rgb, meta=read_photo(p)
            self.assertEqual(rgb.shape[:2],(1200,900))
            candidates,_,info=intrinsics_for_photo(meta)
            self.assertEqual(len(candidates),3)
            self.assertTrue(info['estimated'])
            self.assertEqual(info['source'],'exif')

    def test_joint_focal_refinement_on_known_geometry(self):
        rng=np.random.default_rng(4)
        xyz=rng.uniform([-1.2,-.8,2.],[1.2,.8,5.],(100,3))
        K=np.array([[700.,0,511.5],[0,700,383.5],[0,0,1]])
        xy=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
        xy+=rng.normal(0,.05,xy.shape)
        initialK=K.copy();initialK[0,0]=initialK[1,1]=740
        T=np.eye(4);T[:3,3]=[.01,.01,-.10]
        result=dict(success=True,inliers=100,pose=pose_dict(T),query_intrinsics=initialK.tolist(),
                    inlier_world_points=xyz.tolist(),inlier_query_pixels=xy.tolist())
        refine_focal(result,(1024,768))
        self.assertTrue(result.get('focal_refined'))
        self.assertLess(abs(result['query_intrinsics'][0][0]-700),2)
        self.assertLess(np.linalg.norm(result['pose']['position_m']),.02)


class HttpIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wifi_localization.app import LocalApp, LocalServer
        data=Path(__file__).resolve().parents[1]/'data/room'
        db=Path(os.environ.get('ROOM_TEST_DB', 'outputs/app_validation/database'))
        if not data.exists() or not (db/'database.json').exists():
            raise unittest.SkipTest('Set ROOM_TEST_DB to an existing room database for real-image HTTP tests')
        cls.temp=tempfile.TemporaryDirectory()
        cls.app=LocalApp(data,cls.temp.name,db_path=db)
        cls.app.start();cls.app.initialization.result(timeout=120)
        if not cls.app.status()['ready']:raise RuntimeError(cls.app.status())
        cls.server=LocalServer(('127.0.0.1',0),cls.app)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.app.close();cls.temp.cleanup()

    def get(self,path):
        return urlopen(self.base+path,timeout=30)

    def post(self,path,data,content_type='application/json',extra=None):
        return urlopen(Request(self.base+path,data=data,headers={'Content-Type':content_type,**(extra or {})}),timeout=30)

    def wait_job(self,id):
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            with self.get(f'/api/jobs/{id}') as response:r=json.load(response)
            if r['status'] in ['done','error']:return r
            time.sleep(.1)
        self.fail('Job did not finish')

    def test_uploaded_photo_to_downloadable_marked_model(self):
        frame=self.app.capture.frames[self.app.capture.ids[17]]
        boundary='room-test-boundary'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nauto\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="phone.jpg"\r\n'
              'Content-Type: image/jpeg\r\n\r\n').encode()+frame.image.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        with self.post('/api/localize',body,f'multipart/form-data; boundary={boundary}') as response:
            self.assertEqual(response.status,202);job=json.load(response)
        job=self.wait_job(job['id']);self.assertEqual(job['status'],'done',job)
        result=job['result'];self.assertTrue(result['success'],result.get('reason'))
        self.assertTrue(result['camera']['estimated'])
        self.assertGreaterEqual(result['excluded_reference_count'],1)
        with self.get(result['urls']['model']) as response:
            glb=response.read();self.assertEqual(glb[:4],b'glTF')
        scene=trimesh.load_scene(io.BytesIO(glb),file_type='glb')
        matches=[name for name in scene.graph.nodes_geometry if 'localized_camera_center' in name]
        self.assertTrue(matches)
        transform,_=scene.graph[matches[0]]
        np.testing.assert_allclose(transform[:3,3],result['pose']['position_m'],atol=1e-6)
        with self.get(result['urls']['viewer']) as response:self.assertIn(b'plotly',response.read().lower())

    def test_phone_photo_regression_with_enhanced_matching(self):
        photo=os.environ.get('ROOM_REGRESSION_PHOTO')
        if not photo or not Path(photo).is_file():
            self.skipTest('Set ROOM_REGRESSION_PHOTO to an independent phone image')
        from wifi_localization.learned import LearnedLocalizer
        cache=os.environ.get('ROOM_TEST_LEARNED_CACHE',str(self.app.output/'learned_features'))
        self.app.learned=LearnedLocalizer(self.app.capture,self.app.db.ids,cache)
        boundary='independent-phone-regression'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nauto\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="independent_phone.jpg"\r\n'
              'Content-Type: image/jpeg\r\n\r\n').encode()+Path(photo).read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
        with self.post('/api/localize',body,f'multipart/form-data; boundary={boundary}') as response:job=json.load(response)
        job=self.wait_job(job['id']);self.assertEqual(job['status'],'done',job.get('message'))
        result=job['result'];self.assertTrue(result['success'],result.get('reason'))
        self.assertIn('LightGlue',result['method'])
        self.assertGreaterEqual(result['inliers'],12)
        self.assertGreaterEqual(result['occupied_grid_cells'],4)
        self.assertGreaterEqual(result['inlier_ratio'],.12)
        self.assertGreaterEqual(result['supporting_references'],2)
        self.assertTrue(result['camera']['estimated'])
        self.assertIn('model',result['urls'])

    def test_saved_profile_and_http_boundaries(self):
        calibration=dict(width=1024,height=768,fx=718,fy=718,cx=512,cy=384)
        with self.post('/api/calibration',json.dumps(calibration).encode()) as response:self.assertTrue(json.load(response)['saved'])
        with self.get('/api/status') as response:self.assertTrue(json.load(response)['has_calibration'])
        with self.get('/') as response:self.assertIn('定位这张照片',response.read().decode())
        with self.assertRaises(HTTPError) as caught:
            self.post('/api/sample',b'{}',extra={'Origin':'http://unrelated.example'})
        self.assertEqual(caught.exception.code,403)
        with self.assertRaises(HTTPError) as caught:self.get('/results/../../README.md')
        self.assertEqual(caught.exception.code,404)
        with self.assertRaises(HTTPError) as caught:self.post('/api/calibration',b'{}')
        self.assertEqual(caught.exception.code,400)

    def test_sample_and_blank_photo_are_distinguished(self):
        with self.post('/api/sample',b'{}') as response:job=json.load(response)
        r=self.wait_job(job['id'])
        self.assertEqual(r['status'],'done',r)
        self.assertTrue(r['result']['success'])
        self.assertTrue(r['result']['sample'])
        buffer=io.BytesIO();Image.new('RGB',(1024,768)).save(buffer,format='PNG')
        boundary='blank-query-test'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="blank.png"\r\n'
              'Content-Type: image/png\r\n\r\n').encode()+buffer.getvalue()+f'\r\n--{boundary}--\r\n'.encode()
        with self.post('/api/localize',body,f'multipart/form-data; boundary={boundary}') as response:job=json.load(response)
        r=self.wait_job(job['id'])
        self.assertEqual(r['status'],'done',r)
        self.assertFalse(r['result']['success'])
        self.assertNotIn('model',r['result']['urls'])


if __name__=='__main__':unittest.main()
