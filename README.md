# M202A Project

The project website is in [`index.html`](index.html). Implemented modules are organized in their own directories.

## Visual localization

[`visual_localization/`](visual_localization/) contains the local visual localization application, including its desktop tools, iPhone project, tests, dependency files, launch scripts, and documentation.

From the repository root, start it with:

```bash
cd visual_localization
bash run_web.command
```

See the [module README](visual_localization/README.md) for setup and usage, and the [reference-data guide](visual_localization/docs/WEB_UPLOAD.md) for supported imports.

An independent, offline iPhone app and Xcode project are in [`visual_localization/ios/`](visual_localization/ios/). See the [iPhone setup guide](visual_localization/ios/README.md) for building, signing, and transferring a room package. Native device validation is still required.

Once dependencies and enhanced-matching weights have been downloaded, the app can be started without running installation again:

```bash
cd visual_localization
.venv/bin/python -m wifi_localization web --output outputs/web
```

Models, reference photos, generated databases, caches, and experiment outputs are supplied or created locally and are not included in this repository.

