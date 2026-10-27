"""Select a reader from the prepared capture format."""
from pathlib import Path
from .polycam import Capture


def load_capture(root, alignment='forward', depth_scale=.001):
    if (Path(root) / 'references.json').is_file():
        from .posed import PosedCapture
        return PosedCapture(root)
    if (Path(root) / 'seven_scenes.json').is_file():
        from .seven_scenes import SevenScenesCapture
        return SevenScenesCapture(root)
    return Capture(root, alignment, depth_scale)
