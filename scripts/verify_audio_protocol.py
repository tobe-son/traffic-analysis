"""Synthetic mechanism check, NOT a traffic recognition experiment."""
import json
from pathlib import Path
import numpy as np
from research.audio_protocol import prepare

rate = 48000
t = np.arange(rate) / rate
rows = []
for frequency in [1000, 7000]:
    x = np.sin(2*np.pi*frequency*t)
    stereo = np.column_stack([x, .25*x])
    filtered = prepare(stereo, rate, 8000)[100:-100]
    naive = stereo[::6][100:-100]
    rms = lambda y: float(np.sqrt(np.mean(y*y)))
    rows.append({'source_hz':frequency, 'target_sample_rate':8000,
                 'naive_left_rms':rms(naive[:,0]),
                 'filtered_left_rms':rms(filtered[:,0]),
                 'right_to_left_rms':rms(filtered[:,1])/rms(filtered[:,0])})
output = Path('docs/research_2026/verification/synthetic_audio.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'kind':'synthetic mechanism verification only', 'results':rows}, indent=2)+'\n')
print(output.read_text())
