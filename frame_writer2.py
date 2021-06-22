# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import ffmpeg
import sys, os, time
Image.MAX_IMAGE_PIXELS = 500000 * 1920

class FFmpegFrameWriter:
    def __init__(self, video_file_name, fps=30, size=(1280, 720), bitrate='10240k'):
        self.video_file_name = video_file_name
        self.fps = fps
        video_width, video_height = size
        self.frame = np.zeros((video_height, video_width, 3), dtype='uint8')
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps, s=f'{video_width}x{video_height}')
            .output(video_file_name, pix_fmt='yuv420p', video_bitrate=bitrate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait()
        except:
            pass

    def add_frame(self):
            self.add(self.frame)

    def add(self, frame):
        self.process.stdin.write(frame.tobytes())
