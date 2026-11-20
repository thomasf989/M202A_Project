#!/usr/bin/env python3
"""Prepare any supported scan for offline import into Room Wi-Fi Explorer."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'visual_localization'))
from wifi_localization.rooms import prepare_room
from wifi_localization.export_iphone import export

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--references',required=True,type=Path)
p.add_argument('--model',required=True,type=Path)
p.add_argument('--name',required=True)
p.add_argument('--work',required=True,type=Path,help='New output folder for this scan; never reuse for different inputs')
a=p.parse_args()
if a.work.exists(): p.error('Choose a new --work folder to avoid stale reference data.')
a.work.mkdir(parents=True)
prepare_room(a.references,a.model,a.work/'data',{})
map_id=export(a.work/'data',a.work/'database',a.work/'room',a.name)
print(f'Import on iPhone: {(a.work/"room.wifiroom").resolve()}\nMap ID: {map_id}')
