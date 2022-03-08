"""false ender

Usage:
    false_ender.py <playlist_url>
"""
from __future__ import unicode_literals
from pathlib import Path
import json

from pprint import pprint
from docopt import docopt
import youtube_dl
from pydub import AudioSegment
import numpy as np
from typing import Iterable

THRESHOLD_RMS = .05
MIN_SONG_DURATION_S = 2*60
MAX_SONG_DURATION_S = 4*60
CHUNK_DURATION = 2000 # ms

work_dir = Path() / "cache"

if not work_dir.exists():
    work_dir.mkdir()



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
    def __init__(self, print):
        self.print = print

    def debug(self, msg):
        self.print(msg)

    def warning(self, msg):
        self.print(msg)

    def error(self, msg):
        self.print(msg)



announcements = {
    '2uX-VZb7rvA',  # Solo notice 1
    'vsQb882KVSo',  # Solo notice 2
    'dIXivJkO7fE',  # Solo notice 3
    'E-ZR65NMJII',  # Soul Warz
    'wkhngBqmuZY',  # Halftime Notice
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


def main():
    arguments = docopt(__doc__)

    playlist_logger = LastMessageLogger()
    ydl_opts = {
        'logger': playlist_logger,
        'extract_flat': True,
        'dump_single_json': 'yes',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        out = ydl.download([arguments['<playlist_url>']])
        assert out == 0
        assert playlist_logger.last_error is None
    playlist_info = json.loads(playlist_logger.last_debug)
    print('playlist metadata downloaded')

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
        'logger': ShitLogger(print=print),
        'progress_hooks': [progress_hook],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for i, v in enumerate(ids):
            mp3_file = work_dir / f'{v}.mp3'
            if mp3_file.exists():
                continue
            out = ydl.download([f'https://www.youtube.com/watch?v={v}'])
            assert out == 0
            assert mp3_file.exists()

    # Find things to complain about
    total_duration_s = 0
    for i, v in enumerate(ids):
        if v in announcements or i==0:
            print()
        print(f'{i+1:02d} {v}  {entries[i]["title"]}')

        mp3_file = work_dir / f'{v}.mp3'
        sound = AudioSegment.from_mp3(mp3_file)
        duration_s = len(sound) // 1000
        total_duration_s += duration_s
        if v in announcements:
            continue

        if duration_s < MIN_SONG_DURATION_S:
            print(f'    song too short. duration is {format_timestamp(duration_s)}. recommended more than {format_timestamp(MIN_SONG_DURATION_S)}')
        if duration_s > MAX_SONG_DURATION_S:
            print(f'    song too long. duration is {format_timestamp(duration_s)}. receommended less than {format_timestamp(MAX_SONG_DURATION_S)}')
        fake_ends = detect_fake_ends(sound)
        for s in fake_ends:
            print(f'    fake end at {format_timestamp(s)}')

    print()
    print(f'total playlist duration: {format_timestamp(s)}')


if __name__ == '__main__':
    main()
