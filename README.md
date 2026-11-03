# Sample Rooms

Sample room assets for the M202A visual localization project. This branch provides room data and download instructions. The application is available on the [main branch](https://github.com/thomasf989/M202A_Project/tree/main).

## Available room: 7-Scenes Fire

Download both files from the [Sample Rooms release](https://github.com/thomasf989/M202A_Project/releases/tag/Sample_rooms), or use the direct links below.

| File | Purpose | Size |
| --- | --- | --- |
| [7Scenes_Fire.glb](https://github.com/thomasf989/M202A_Project/releases/download/Sample_rooms/7Scenes_Fire.glb) | 3D model of the Fire scene | 5.9 MB |
| [7Scenes_Fire_Reference_Data.zip](https://github.com/thomasf989/M202A_Project/releases/download/Sample_rooms/7Scenes_Fire_Reference_Data.zip) | Reference-data package for importing the room into the localization application | 68.0 MB |

Download the two named assets. GitHub's automatically generated **Source code (zip)** and **Source code (tar.gz)** archives are snapshots of the repository, and do not include the separately uploaded release attachments.

## Import the room

1. Download `7Scenes_Fire.glb` and `7Scenes_Fire_Reference_Data.zip`.
2. Start the localization application. For the repository version, follow the [application setup instructions](https://github.com/thomasf989/M202A_Project/blob/main/README.md).
3. Open **Add Room**.
4. Select `7Scenes_Fire.glb` as the room model and `7Scenes_Fire_Reference_Data.zip` as the reference-data package. Upload the ZIP directly without extracting it.
5. Submit the room and wait for the application to finish preparing the reference database.
6. Select the imported room and upload a query photo for localization.

The GLB provides the 3D scene geometry. The reference package supplies the reference data used to build the localization database. Use the matching model and reference package together, and preserve their coordinate alignment.

For supported formats and import options, see the [reference-data guide](https://github.com/thomasf989/M202A_Project/blob/main/visual_localization/docs/WEB_UPLOAD.md).

## Future rooms

Additional rooms can be listed here with links to their matching model and reference-data packages.
