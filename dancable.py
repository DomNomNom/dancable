"""false ender

Usage:
    false_ender.py <playlist_url>
"""
from __future__ import unicode_literals
from pathlib import Path
import json
from typing import Iterable

from pprint import pprint
from docopt import docopt
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import numpy as np

THRESHOLD_RMS = .05
MIN_SONG_DURATION_S = 2*60
MAX_SONG_DURATION_S = 4*60
CHUNK_DURATION = 2000 # ms

work_dir = Path() / "cache"

if not work_dir.exists():
    work_dir.mkdir()

# # https://stackoverflow.com/questions/14630288/unicodeencodeerror-charmap-codec-cant-encode-character-maps-to-undefined
import sys
# import codecs
# if sys.stdout.encoding != 'cp850':
#     print('aaaaasfasfasf', sys.stdout.encoding)
#     sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
# if sys.stderr.encoding != 'cp850':
#     sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

class LastMessageLogger:
    def __init__(self):
        self.last_debug = None
        self.last_warning = None
        self.last_error = None

    def debug(self, msg):
        self.last_debug = msg

    def warning(self, msg):
        self.last_warning = msg
        print(msg)

    def error(self, msg):
        self.last_error = msg
        print(msg)

class ShitLogger:

    def debug(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)



specials = {
    '2uX-VZb7rvA': 'solo',  # Solo notice 1
    'vsQb882KVSo': 'solo',  # Solo notice 2
    'dIXivJkO7fE': 'solo',  # Solo notice 3
    'E-ZR65NMJII': 'solo',  # Soul Warz
    'wkhngBqmuZY': 'halftime',  # Halftime Notice
}

def detect_fake_ends(sound: AudioSegment) -> Iterable[int]:
    """Returns timestamps when a fake end begins"""
    last_chunk_was_quiet = True
    last_fake_end = None
    for start_ms in range(0, len(sound), CHUNK_DURATION):
        # don't process last chunk
        if start_ms+CHUNK_DURATION >= len(sound):
            continue
        chunk = sound[start_ms:start_ms+CHUNK_DURATION]
        samples = [s.get_array_of_samples() for s in chunk.split_to_mono()]
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        rms = np.sqrt(np.mean(fp_arr**2))
        is_quet = rms < THRESHOLD_RMS
        if not last_chunk_was_quiet and is_quet:# and (last_fake_end is not None and s > last_fake_end+10):
            s = start_ms//1000
            if s > 60 and start_ms < len(sound)*.8:
                last_fake_end = s
                yield s
            last_chunk_was_quiet = True
        elif last_chunk_was_quiet and not is_quet:
            last_chunk_was_quiet = False

def format_timestamp(seconds: float) -> str:
    return f'{seconds//60}:{int(seconds)%60:02d}'

def dancable(playlist_url):
    playlist_logger = LastMessageLogger()
    ydl_opts = {
        'logger': playlist_logger,
        'extract_flat': True,
        'dump_single_json': 'yes',
        'source_address': '0.0.0.0',
    }
    with YoutubeDL(ydl_opts) as ydl:
        out = ydl.download([playlist_url])
        assert out == 0
        assert playlist_logger.last_error is None
    playlist_info = json.loads(playlist_logger.last_debug)
    print('playlist metadata downloaded.')

    entries = playlist_info['entries']
    ids = [e['id'] for e in entries]


    # download all the things.
    def progress_hook(d):
        if d.get('status') == 'downloading':
            download_ratio = d.get('downloaded_bytes', 0) / d.get('total_bytes', 1)
            print(f"[download {i:02d}/{len(entries)}] {download_ratio*100:02.2f}% of {d.get('total_bytes', 0)/1000**2 :.1f}MB")
        elif d.get('status') == 'finished':
            pass
        else:
            print(d)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(work_dir / '%(id)s.%(ext)s'),
        # 'nooverwrites': True,
        'noprogress': True,
        'logger': ShitLogger(),
        'progress_hooks': [progress_hook],
        'source_address': '0.0.0.0',
    }
    with YoutubeDL(ydl_opts) as ydl:
        for i, v in enumerate(ids):
            mp3_file = work_dir / f'{v}.mp3'
            if mp3_file.exists():
                continue
            out = ydl.download([f'https://www.youtube.com/watch?v={v}'])
            assert out == 0
            assert mp3_file.exists()
    print('playlist downloaded.')

    # Find things to complain about
    total_duration_s = 0
    for i, v in enumerate(ids):
        if v in specials or i==0:
            print()
        print(f"{i+1:02d} {entries[i]['title'].encode(sys.stdout.encoding, errors='replace').decode()}")

        mp3_file = work_dir / f'{v}.mp3'
        sound = AudioSegment.from_mp3(mp3_file)
        duration_s = len(sound) // 1000
        total_duration_s += duration_s
        if v in specials:
            continue

        if duration_s < MIN_SONG_DURATION_S:
            print(f'    song too short. got {format_timestamp(duration_s)}, want >{format_timestamp(MIN_SONG_DURATION_S)}')
        if duration_s > MAX_SONG_DURATION_S and (i == 0 or specials.get(ids[i-1]) == 'solo'):
            print(f'    song too long. got {format_timestamp(duration_s)}, want <{format_timestamp(MAX_SONG_DURATION_S)}')
        fake_ends = detect_fake_ends(sound)
        for s in fake_ends:
            print(f'    fake end at {format_timestamp(s)}')

    print()
    print(f'total playlist duration: {format_timestamp(total_duration_s)}')

def main():
    arguments = docopt(__doc__)
    dancable(arguments['<playlist_url>'])

if __name__ == '__main__':
    main(
