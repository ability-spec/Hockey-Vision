"""CPU-only CLI wiring tests with model and subprocess doubles."""
import ast
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]

def functions(*names, **scope):
    tree = ast.parse((ROOT / 'run_models.py').read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), 'run_models.py', 'exec'), scope)
    return scope

@pytest.mark.parametrize('device,extension', [('cpu','.pt'), ('mps','.pt'), ('cuda','.engine'), ('0','.engine')])
def test_device_selects_compatible_weights(tmp_path, device, extension):
    (tmp_path / 'player_model.engine').touch()
    scope = functions('model_path', MODELS_DIR=tmp_path, pick_device=lambda device: device)
    assert scope['model_path']('player', device).suffix == extension

@pytest.mark.parametrize('opened', [False, True])
def test_unreadable_video_raises_without_overwriting_previous_results(tmp_path, opened):
    class Capture:
        released = False
        def isOpened(self): return opened
        def read(self): return False, None
        def get(self, _): return 30
        def release(self): self.released = True
    cap = Capture()
    cv2 = SimpleNamespace(VideoCapture=lambda _: cap, VideoWriter_fourcc=lambda *a: 0,
                          CAP_PROP_FPS=0, CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FRAME_COUNT=3)
    scope = functions('process_video', cv2=cv2, JerseyVotes=lambda: None, json=json)
    old = tmp_path / 'detections.json'; old.write_text('old')
    args = SimpleNamespace(write_video=False, max_frames=None)
    with pytest.raises(RuntimeError): scope['process_video'](Path('bad.mp4'), {}, args, tmp_path)
    assert cap.released
    assert old.read_text() == 'old'

@pytest.mark.parametrize('args,root', [([], 'outputs'), (['--output','custom folder'], 'custom folder'), (['--output=another'], 'another')])
def test_wrapper_propagates_output_root(tmp_path, args, root):
    fake = tmp_path / 'python'
    fake.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
if args[0]=='run_models.py':
    root='outputs'
    for i,arg in enumerate(args):
        if arg=='--output': root=args[i+1]
        if arg.startswith('--output='): root=arg.split('=',1)[1]
    path=pathlib.Path(root)/pathlib.Path(args[1]).stem/'detections.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[]')
else:
    pathlib.Path('called.json').write_text(json.dumps(args))
''')
    fake.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}
    result = subprocess.run(['bash', str(ROOT/'jetson/run_clip.sh'), 'clip.mp4', *args], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path/'called.json').read_text())[1] == f'{root}/clip/detections.json'

@pytest.mark.parametrize('exit_code', [0, 1])
def test_wrapper_never_passes_stale_detections(tmp_path, exit_code):
    path=tmp_path/'outputs/clip/detections.json'; path.parent.mkdir(parents=True); path.write_text('old')
    os.utime(path, (1, 1))
    fake=tmp_path/'python'; fake.write_text(f'#!/bin/sh\nexit {exit_code}\n'); fake.chmod(0o755)
    env={**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH']}
    result=subprocess.run(['bash', str(ROOT/'jetson/run_clip.sh'), 'clip.mp4'], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode != 0
