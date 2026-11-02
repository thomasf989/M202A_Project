# Web Upload and Automatic Reference-Library Creation

On macOS, run `bash run_web.command` or `bash run_app.command`. Your browser will automatically open the local web page.
An internet connection is required to install dependencies the first time. Keep the terminal window running and press Control+C to exit.
`bash run_7scenes.command` also opens this page with Chess selected by default.

## Instructions

Click “Add Room,” select a model GLB and reference-data ZIP, and then click “Upload and Prepare Room.”
After the upload finishes, the program automatically checks the format and camera information, reads the model, extracts reference features, and builds the localization reference library.
When preparation is complete, the dialog closes. You can then select a new query photo to localize or try a sample photo.
Results include the metric position in the GLB coordinate system, camera direction, a colored GLB with a camera marker, pose JSON, and an offline 3D page.

Added rooms are saved locally and remain available the next time the program starts. Switching rooms also switches the reference library and query-camera settings.
After switching, downloads from earlier jobs still correspond to their original rooms. Query photos are not added to the reference library automatically.

## Polycam

Upload a raw export ZIP and GLB from the same scan. The ZIP must contain `optimized_roomplan.json`,
`keyframes/corrected_images/`, `corrected_cameras/`, `depth/`, and `confidence/`.
Reference-camera parameters, poses, and the alignment matrix are read from the exported data. Sampled depth is checked against the model for consistency, and models with clearly mismatched coordinates are rejected.
Preserve the original model coordinates and metric scale. An ordinary photo ZIP is not the same as a raw Polycam export ZIP.

## 7-Scenes

The program supports an official outer scene ZIP, a single scene that has been extracted and repackaged, and reference packages prepared by this program.
Raw ZIPs are divided according to the official `TrainSplit.txt` / `TestSplit.txt`; test images are not added to the reference library.
By default, raw training data is sampled every 10 frames and test data every 50 frames. These values can be changed under Advanced Options.
A prepared package includes `seven_scenes.json`; its frame list, intrinsics, and coordinate settings are preserved without resampling.

The default model convention for official raw data matches the Y-up GLB provided by this project: world coordinates are left-multiplied by `diag(1,-1,-1,1)`.
If your GLB preserves the dataset's original world coordinates, select the corresponding Advanced Options setting. You may also enter a custom 4×4 rigid transformation matrix.
The raw RGB and depth data are not fully calibrated. The adapter uses depth rendered from the model on the RGB grid, and the default RGB intrinsics are approximate.
The web page allows reference-only packages for real localization; command-line evaluation imports still require independent test frames.
If you have only a few raw reference images, set the sampling interval to 1 to ensure that at least four reference images are available.

## Posed Photos for Any Room

Place the reference photos and `references.json` in the same ZIP:

```text
references.json
images/photo-001.jpg
images/photo-002.jpg
...
```

You can download the `references.json` structure from the web page or view `wifi_localization/web/reference-format.json`.
Its fields are:

| Field | Meaning |
|---|---|
| `format` | Fixed value: `posed-images-v1` |
| `intrinsics` | Default width, height, fx, fy, cx, and cy; these may be overridden per frame |
| `camera_convention` | `opencv`: X right, Y down, Z forward; or `opengl`: X right, Y up, Z backward |
| `world_to_glb` | A 4×4 rigid transformation from the world coordinates used by the reference poses to the uploaded GLB; defaults to the identity matrix |
| `frames[].id` | Unique frame ID containing only letters, numbers, underscores, or hyphens |
| `frames[].image` | Relative photo path inside the ZIP |
| `frames[].camera_to_world` | Column-vector-convention 4×4 camera-to-world matrix with position measured in meters |
| `frames[].split` | `reference`/`train` enters the reference library; `test` remains held out; defaults to `reference` |

At least four reference photos are required. Photos must be undistorted and stored in the pixel orientation corresponding to the intrinsics and poses; EXIF rotation is not used.
Reference-image depth is rendered from the model, so separate sensor depth is not required. The model itself must be aligned correctly with the reference poses.
For new photos taken with a different phone or lens, use automatic focal-length estimation or import your own query-camera parameters; do not assume they match the reference camera.
The camera values and positions in the template are format examples only and must be replaced with real data.

## Uploads, Storage, and Performance

The maximum GLB size is 512 MB, the maximum reference ZIP size is 4 GB, and expanded data is limited to 12 GB. Files are transferred in 4 MB chunks, so large files are not cached in full within a request.
After an interruption, select the same file again to resume from the previous offset. Upload records are retained in the local browser and program directory.
The program reuses an existing room reference library when the model, ZIP, and import options are identical.
Nested 7-Scenes sequences use temporary disk files to avoid loading each 500 MB sequence ZIP entirely into memory.
Building the library is a one-time task for each room; subsequent photos reuse the cache. Enhanced matching weights may still be downloaded from the internet on first use.

New rooms are saved under `outputs/web/rooms/<id>/data`, with the feature library and prediction results under its `output/` directory.
This is a single-user local web page, not a deployed public service. Room data and localization calculations remain on the computer running the program.
