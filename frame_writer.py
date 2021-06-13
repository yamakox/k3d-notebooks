# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import ffmpeg
import sys, os, time
Image.MAX_IMAGE_PIXELS = 500000 * 1920
if sys.platform != 'win32':
    import fcntl

class FFmpegFrameWriter:
    def __init__(self, video_file_name, fps=30, size=(1280, 720), bitrate='10240k', stdout=False):
        self.video_file_name = video_file_name
        self.fps = fps
        self.stdout = stdout
        video_width, video_height = size
        self.frame = np.zeros((video_height, video_width, 3), dtype='uint8')
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps, s=f'{video_width}x{video_height}')
            .output(video_file_name, pix_fmt='yuv420p', video_bitrate=bitrate)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        if self.stdout:
            if sys.platform != 'win32':
                fl =  fcntl.fcntl(self.process.stderr, fcntl.F_GETFL)
                fcntl.fcntl(self.process.stderr, fcntl.F_SETFL, fl|os.O_NONBLOCK)
            else:
                self.stdout = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait()
            if self.stdout:
                print()
        except:
            pass

    def add_frame(self):
            self.add(self.frame)

    def add(self, frame):
        self.process.stdin.write(frame.tobytes())
        if not self.stdout:
            return
        if self.process.stderr.readable():
            data = self.process.stderr.read()
            if data:
                print(data.decode('utf-8'), end='')
            sys.stdout.flush()
