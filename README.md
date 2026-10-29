# M202A Project

The project website is in [`index.html`](index.html). Implemented modules are organized in their own directories.

## Visual localization

[`visual_localization/`](visual_localization/) contains the v1.4 local visual localization application, including its Python package, web interface, tests, dependency files, launch scripts, and documentation.

From the repository root, start it with:

```bash
cd visual_localization
bash run_web.command
```

See the [module README](visual_localization/README.md) for setup and usage, and the [reference-data guide](visual_localization/docs/WEB_UPLOAD.md) for supported imports.

Once dependencies and enhanced-matching weights have been downloaded, the app can be started without running installation again:

```bash
cd visual_localization
.venv/bin/python -m wifi_localization web --output outputs/web
```

Models, reference photos, generated databases, caches, and experiment outputs are supplied or created locally and are not included in this repository.

## iPhone walking baseline

[`ios/WalkingBaseline/`](ios/WalkingBaseline/) contains a native ARKit/RealityKit
marker-aligned walking prototype, a printable reference image, XcodeGen project
configuration, and alignment unit tests. See its README for Mac build steps and
the physical-device acceptance checklist. This baseline does not yet integrate
Wi-Fi measurements or automatic visual-localization corrections.
