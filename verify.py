#!/usr/bin/env python3
"""Verify the packaged file inventory without importing experiment dependencies."""
import argparse, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
m=json.loads((root/'MANIFEST.json').read_text());bad=[]
for row in m['files']:
 p=root/row['path']
 if not p.is_file() or p.stat().st_size!=row['bytes']:
  bad.append(row['path']);continue
 if args.full:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  if h.hexdigest()!=row['sha256']:bad.append(row['path'])
print(f"{len(m['files'])} files checked; {len(bad)} mismatches")
for p in bad[:30]:print(p)
raise SystemExit(bool(bad))
