import argparse
import re
import subprocess

import av
import cv2
import numpy as np
from decord import VideoReader, cpu
from torchvision import io

parser = argparse.ArgumentParser(description='Count frames.')
parser.add_argument('file', type=str, help='which file')
args = parser.parse_args()

video_path = args.file


def frame_count_ffprobe_query(filename):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of',
        'default=nokey=1:noprint_wrappers=1', filename
    ],
                            stdout=subprocess.PIPE,
                            encoding='utf8',
                            check=True)
    return result.stdout.strip().strip()


def frame_count_ffmpeg_count(filename):
    result = subprocess.run(['ffmpeg', '-i', filename, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-'],
                            stderr=subprocess.PIPE,
                            encoding='utf8',
                            check=True)
    m = re.search(r'frame=\s*(\d+) ', result.stderr)
    return m.groups()[0]


def frame_count_ffmpeg_decode(filename):
    result = subprocess.run(['ffmpeg', '-i', filename, '/dev/shm/img-%5d.jpg'],
                            stderr=subprocess.PIPE,
                            encoding='utf8',
                            check=True)
    m = re.findall(r'frame=\s*(\d+) ', result.stderr)
    return m[-1]


def frame_count_denseflow_rgb(filename):
    result = subprocess.run([
        'srun', '--partition=pat_mars', '--gres=gpu:1', '-n1', '--ntasks-per-node=1', '--job-name=ha', '-x',
        'SH-IDC1-10-198-4-108', '--kill-on-bad-exit=1', 'denseflow', '-o=/dev/shm/', filename
    ],
                            stdout=subprocess.PIPE,
                            encoding='utf8',
                            check=True)
    m = re.search(r'(\d+) frames', result.stdout)
    return m.groups()[0]


def frame_count_denseflow_flow(filename):
    result = subprocess.run([
        'srun', '--partition=pat_mars', '--gres=gpu:1', '-n1', '--ntasks-per-node=1', '--job-name=ha', '-x',
        'SH-IDC1-10-198-4-108', '--kill-on-bad-exit=1', 'denseflow', '-s=1', '-st=png', '-o=/dev/shm/', filename
    ],
                            stdout=subprocess.PIPE,
                            encoding='utf8',
                            check=True)
    m = re.search(r'(\d+) frames', result.stdout)
    return m.groups()[0]


def frame_count_cv2_query(filename):
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def frame_count_cv2_count(filename):
    cap = cv2.VideoCapture(filename)
    n = 0
    has_next, _ = cap.read()
    while has_next:
        n += 1
        has_next, _ = cap.read()
    return n


def frame_count_decord(filename):
    vr = VideoReader(video_path, ctx=cpu(0))
    return len(vr)


def frame_count_pyav(filename):
    container = av.open(video_path)
    return len(list(container.decode(video=0)))


def frame_count_torchvision(filename):
    vframes, _, _ = io.read_video(filename, pts_unit='sec')
    return len(vframes)


print(f'ffprobe query frames {frame_count_ffprobe_query(video_path)}')
print(f'ffmpeg count frames {frame_count_ffmpeg_count(video_path)}')
print(f'ffmpeg decode frames {frame_count_ffmpeg_decode(video_path)}')
print(f'deseflow rgb frames {frame_count_denseflow_rgb(video_path)}')
print(f'deseflow flow frames {frame_count_denseflow_flow(video_path)}')
print(f'cv2 query frames {frame_count_cv2_query(video_path)}')
print(f'cv2 count frames {frame_count_cv2_count(video_path)}')
print(f'decord frames {frame_count_decord(video_path)}')
print(f'pyav frames {frame_count_pyav(video_path)}')
print(f'torchvision frames {frame_count_torchvision(video_path)}')
