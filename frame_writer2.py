# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import ffmpeg
import sys, os, time, tempfile
Image.MAX_IMAGE_PIXELS = 500000 * 1920

class FFmpegFrameWriter:
    def __init__(self, video_file_name, fps=30, size=(1280, 720), bitrate='10240k'):
        video_width, video_height = size
        self.frame = np.zeros((video_height, video_width, 3), dtype='uint8')
        self.video_file_name = video_file_name
        self.fps = fps
        self.bitrate = bitrate
        self.tempdir = tempfile.TemporaryDirectory(dir='.')
        self.frame_number = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        (
            ffmpeg
            .input(os.path.join(self.tempdir.name, '*.png'), pattern_type='glob', framerate=self.fps)
            .output(self.video_file_name, pix_fmt='yuv420p', video_bitrate=self.bitrate)
            .overwrite_output()
            .run(capture_stdout=True)
        )
        self.tempdir.cleanup()

    def add_frame(self):
        self.add(self.frame)

    def add(self, frame):
        png = Image.fromarray(frame)
        png.save(os.path.join(self.tempdir.name, f'{self.frame_number}.png'.zfill(10)))
        self.frame_number += 1


obsoleted_skvideo = r'''
import skvideo.io
class FFmpegFrameWriter:
    def __init__(self, video_file_name, fps=30, size=(1280, 720), bitrate='10240k'):
        video_width, video_height = size
        self.frame = np.zeros((video_height, video_width, 3), dtype='uint8')
        self.writer = skvideo.io.FFmpegWriter(
            video_file_name, 
            outputdict = {
                '-vcodec': 'libx264', 
                '-b': bitrate, 
                '-pix_fmt': 'yuv420p', 
                '-r': str(fps), 
                '-s': f'{video_width}x{video_height}', 
            }, 
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        try:
            self.writer.close()
        except:
            pass

    def add_frame(self):
            self.add(self.frame)

    def add(self, frame):
        self.writer.writeFrame(frame)
'''
