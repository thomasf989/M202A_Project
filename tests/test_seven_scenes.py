"""Dataset split, coordinates, RGB-depth registration and browser contracts."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image
import trimesh
from wifi_localization.capture import load_capture
from wifi_localization.seven_scenes import prepare_seven_scenes, DEFAULT_WORLD_TO_GLB, dataset_pose
from wifi_localization.features import build_database, Database
from wifi_localization.polycam import read_json, write_json
from wifi_localization.raster import _depth
from wifi_localization.visualization import load_mesh


class SevenScenesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        source=self.root/'source';source.mkdir();self.source=source
        (source/'TrainSplit.txt').write_text('sequence1\n');(source/'TestSplit.txt').write_text('sequence3\n')
        rng=np.random.default_rng(5)
        for seq,count in [('seq-01',4),('seq-03',1)]:
            (source/seq).mkdir()
            for i in range(count):
                base=source/seq/f'frame-{i:06d}'
                Image.fromarray(rng.integers(0,256,(480,640,3),dtype=np.uint8)).save(str(base)+'.color.png')
                # Deliberately invalid sensor depth: the adapter must use the
                # RGB-aligned mesh, rather than raw-depth pixel indexing.
                Image.fromarray(np.full((480,640),65535,np.uint16)).save(str(base)+'.depth.png')
                np.savetxt(str(base)+'.pose.txt',np.eye(4))
        m=trimesh.Trimesh([[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],[[0,1,2],[0,2,3]],process=False)
        m.visual.vertex_colors=np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255],[255,255,0,255]])
        m.apply_transform(DEFAULT_WORLD_TO_GLB);self.glb=self.root/'room.glb';m.export(self.glb)
        self.data=prepare_seven_scenes(source,self.glb,self.root/'prepared',1,1)

    def tearDown(self):self.temp.cleanup()

    def test_metric_axes_and_mesh_depth_without_registered_sensor_depth(self):
        c=load_capture(self.data);f=c.frames[c.reference_ids[0]]
        xyz,valid=c.world_points(f,[[320,240],[420,240],[-1,0]])
        np.testing.assert_allclose(xyz[:2],[[0,0,-2],[200/525,0,-2]],atol=1e-6)
        self.assertEqual(valid.tolist(),[True,True,False])
        np.testing.assert_allclose(f.T_world_cv,DEFAULT_WORLD_TO_GLB)
        mesh=load_mesh(c.glb)
        self.assertGreater(len(np.unique(mesh.visual.vertex_colors,axis=0)),1)

    def test_perspective_depth_and_occlusion(self):
        K=np.array([[10.,0,10],[0,10,10],[0,0,1.]])
        # First plane is tilted, second plane is behind it. At pixel (10,10)
        # the tilted surface intersects the optical axis at z=2 exactly.
        near=np.array([[-2.,-2.,1.],[2,-2,3],[0,2,2]])
        far=near.copy();far[:,2]+=4
        vertices=np.vstack([near,far]);faces=np.array([[3,4,5],[0,1,2]])
        image=_depth(vertices,faces,K,21,21,.05,10.)
        self.assertAlmostEqual(float(image[10,10]),2.,places=5)

    def test_reference_database_never_includes_official_test_frames(self):
        c=load_capture(self.data)
        meta=build_database(c,self.root/'db',query_every=None,nfeatures=80,words=16)
        self.assertEqual(set(meta['reference_ids']),set(c.reference_ids))
        self.assertEqual(meta['query_ids'],c.query_ids)
        self.assertFalse(set(meta['query_ids'])&set(meta['reference_ids']))
        self.assertFalse((self.root/'db/features'/f'{c.query_ids[0]}.npz').exists())
        self.assertIn('SEQUENCES',meta['split'])
        Database(self.root/'db',c)
        meta['reference_ids'].append(c.query_ids[0]);write_json(self.root/'db/database.json',meta)
        with self.assertRaisesRegex(ValueError,'official sequence split'):Database(self.root/'db',c)

    def test_overlapping_sequences_and_stale_mesh_are_rejected(self):
        c=load_capture(self.data);before=c.fingerprint()
        changed=trimesh.load(self.data/'room.glb',force='scene');changed.apply_translation([.01,0,0]);changed.export(self.data/'room.glb')
        self.assertNotEqual(before,load_capture(self.data).fingerprint())
        manifest=read_json(self.data/'seven_scenes.json');manifest['test_sequences']=['seq-01'];write_json(self.data/'seven_scenes.json',manifest)
        with self.assertRaises(ValueError):load_capture(self.data)

    def test_preview_without_test_sequence_does_not_become_a_benchmark(self):
        (self.source/'TestSplit.txt').write_text('sequence5\n')
        with self.assertRaisesRegex(ValueError,'official test frame'):
            prepare_seven_scenes(self.source,self.glb,self.root/'missing_test',1,1)
        self.assertFalse((self.root/'missing_test/seven_scenes.json').exists())

    def test_official_nested_zip_layout_and_unrelated_archive_path(self):
        archive=self.root/'chess.zip'
        with zipfile.ZipFile(archive,'w') as outer:
            for name in ['TrainSplit.txt','TestSplit.txt']:outer.writestr('chess/'+name,(self.source/name).read_bytes())
            for seq in ['seq-01','seq-03']:
                buffer=io.BytesIO()
                with zipfile.ZipFile(buffer,'w') as inner:
                    for p in (self.source/seq).iterdir():inner.writestr(seq+'/'+p.name,p.read_bytes())
                    inner.writestr('../unexpected.txt','not an input frame')
                outer.writestr('chess/'+seq+'.zip',buffer.getvalue())
        prepared=prepare_seven_scenes(archive,self.glb,self.root/'nested',1,1)
        c=load_capture(prepared)
        self.assertEqual(len(c.reference_ids),4);self.assertEqual(len(c.query_ids),1)
        self.assertFalse((self.root/'unexpected.txt').exists())
        self.assertEqual(c.frames[c.query_ids[0]].image.read_bytes(),(self.source/'seq-03/frame-000000.color.png').read_bytes())

    def test_small_kinect_rotation_drift_is_corrected_but_reflection_is_rejected(self):
        T=np.eye(4);T[0,0]=.9999;T[0,3]=.3
        corrected=dataset_pose(T,'small drift')
        np.testing.assert_allclose(corrected,np.array([[1,0,0,.3],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        T[0,0]=-1
        with self.assertRaises(ValueError):dataset_pose(T,'reflection')


class SevenScenesRealAppTests(unittest.TestCase):
    def test_known_test_png_upload_and_sample_export(self):
        data=os.environ.get('SEVEN_TEST_DATA');db=os.environ.get('SEVEN_TEST_DB')
        if not data or not db:self.skipTest('Set SEVEN_TEST_DATA and SEVEN_TEST_DB for real-image browser backend validation')
        from wifi_localization.app import LocalApp
        from wifi_localization.localization import pose_errors
        with tempfile.TemporaryDirectory() as d:
            app=LocalApp(data,d,db)
            try:
                app.start();app.initialization.result(timeout=120)
                self.assertTrue(app.status()['ready'],app.status())
                self.assertEqual(app.status()['dataset_format'],'7-scenes')
                # Same upload entry point used by the browser. No reference
                # image, query pose, or query depth is passed to the solver.
                f=app.capture.frames[app.capture.query_ids[0]]
                job=app.submit(f.image.read_bytes(),f.image.name,mode='dataset')
                app.worker.submit(lambda:None).result(timeout=240)
                result=app.job(job['id'])
                self.assertEqual(result['status'],'done',result)
                result=result['result'];self.assertTrue(result['success'],result.get('reason'))
                self.assertLess(result['position_error_m'],.25)
                self.assertIn('model',result['urls'])
                self.assertEqual(result['camera']['source'],'7-scenes')
                exported=trimesh.load_scene(Path(d)/'queries'/job['id']/'localized_room.glb')
                np.testing.assert_allclose(exported.graph['localized_camera_center'][0][:3,3],result['pose']['position_m'])
                displayed=load_mesh(Path(d)/'queries'/job['id']/'localized_room.glb')
                visual=displayed.visual.to_color() if hasattr(displayed.visual,'to_color') else displayed.visual
                self.assertGreater(len(np.unique(visual.vertex_colors,axis=0)),10)
                sample=app.submit(sample=True)
                app.worker.submit(lambda:None).result(timeout=240)
                sample=app.job(sample['id'])
                self.assertEqual(sample['status'],'done',sample)
                self.assertIn('ground_truth',sample['result'])
                self.assertNotIn(app.capture.query_ids[0],app.db.ids)
                # Dataset K is never silently applied to a phone image.
                buf=io.BytesIO();Image.new('RGB',(800,600)).save(buf,format='PNG')
                bad=app.submit(buf.getvalue(),'phone.png',mode='dataset')
                app.worker.submit(lambda:None).result(timeout=30)
                self.assertEqual(app.job(bad['id'])['status'],'error')
            finally:app.close()


if __name__=='__main__':unittest.main()
