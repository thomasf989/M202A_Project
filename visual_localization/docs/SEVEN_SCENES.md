# 7-Scenes import

Supply a scene archive and a GLB model aligned to the same scene. Dataset images, models, precomputed databases, and evaluation results are not included in this source repository.

## Web app

Run `bash run_web.command`, select Add room, and upload the model and reference ZIP. The importer accepts the official outer scene ZIP, extracted scene data repacked as a ZIP, or a prepared archive containing `seven_scenes.json`.

Training and test frames follow `TrainSplit.txt` and `TestSplit.txt`. Only reference frames enter the feature database. A reference-only ZIP can be used for localization; command-line evaluation preparation requires independent test frames.

Upload `.color.png` query images. Raw `.depth.png` files are not query photographs. The dataset camera mode expects 640 by 480 pixels and uses approximate RGB intrinsics. Other cameras should use automatic intrinsics or a calibrated profile.

## Command line

```bash
python -m wifi_localization prepare-7scenes \
  --source /path/to/scene.zip \
  --glb /path/to/scene.glb \
  --output data/seven_scenes/scene \
  --train-stride 10 --test-stride 50

python -m wifi_localization app \
  --data data/seven_scenes/scene \
  --output outputs/scene
```

The source can also be an extracted directory containing split files and `seq-XX/` directories or nested sequence ZIPs. Import into an empty output directory.

## Coordinate and depth conventions

By default, the importer transforms dataset world coordinates with `G = diag(1,-1,-1,1)` into a Y-up GLB frame. This requires a model transformed with the same convention. If the GLB retains the original world frame, provide an identity matrix with `--world-to-glb transform.json`; otherwise supply the known 4 by 4 rigid transform. Models use meters.

OpenCV camera axes are X right, Y down, and Z forward. The camera-to-world transform is `T_GLB_CV = G @ T_dataset_CV`.

The adapter renders reference depth from the supplied mesh on the RGB image grid instead of treating unregistered sensor depth as RGB-aligned. The default `fx=fy=525, cx=320, cy=240` is approximate, not a measured calibration. `--rgb-focal` changes the imported focal length. Raw depth is stored in millimeters and 65535 denotes invalid depth.

A mesh reconstructed using test sequences does not represent a map built exclusively from training sequences. Account for the map source and camera calibration when interpreting evaluation results.

## Dataset sources

- [Microsoft Research 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Chess archive](https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip)
- [TSDF models](https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/tsdf.zip)

Follow the license distributed with the dataset and cite its associated publications when reporting results.
