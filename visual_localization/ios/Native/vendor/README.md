# Vendored native dependencies

These unmodified headers are compiled only by `RoomBuilder.cpp`. No network requests are made by the app.

- [cgltf.h](https://github.com/jkuhlmann/cgltf/tree/v1.15), revision `v1.15`, SHA-256 `e378a21c084bf1f288bb799de827bb26906efb024255f1ecf1705ea13f11c6ec`.
- [stb_image.h](https://github.com/nothings/stb/tree/003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5), revision `003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5`, SHA-256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`.
- [picosha2.h](https://github.com/okdshin/PicoSHA2/tree/27fcf6979298949e8a462e16d09a0351c18fcaf2), revision `27fcf6979298949e8a462e16d09a0351c18fcaf2`, SHA-256 `8f183eaae529cd9d6a3d4843c7559e2a3e3d68b6caaa223e7c24c3c899b3d988`.

All three headers use the MIT license (stb_image also offers a public-domain alternative). Copyright notices remain in the headers; the app bundles `ThirdPartyNotices.txt`.
