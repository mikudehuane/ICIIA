import cv2
import numpy as np


class VideoLoader(object):
    def __init__(
            self,
            train: bool,
            *, resize_height=128, resize_width=171, crop_size=112,
            extract_freq=4, min_frame_count=16
    ):
        self.train = train
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.extract_freq = extract_freq
        self.min_frame_count = min_frame_count

    def __call__(self, video_path):
        frames = self._load_video(video_path)
        frames = frames.astype(np.float32)

        frames = self._crop_video(frames)

        if self.train and np.random.random() < 0.5:
            self._horizontal_flip_(frames)

        self._normalize_(frames)

        frames = frames.transpose((3, 0, 1, 2))
        return frames

    @staticmethod
    def _normalize_(frames):
        for i, frame in enumerate(frames):
            frame -= np.array([[[90.0, 98.0, 102.0]]])

        return frames

    @staticmethod
    def _horizontal_flip_(frames):
        for i, frame in enumerate(frames):
            frames[i] = cv2.flip(frames[i], flipCode=1)

    def _crop_video(self, frames):
        if self.train:
            time_index = np.random.randint(frames.shape[0] - self.min_frame_count)

            height_index = np.random.randint(frames.shape[1] - self.crop_size)
            width_index = np.random.randint(frames.shape[2] - self.crop_size)
        else:
            time_index = (frames.shape[0] - self.min_frame_count) // 2
            height_index = (frames.shape[1] - self.crop_size) // 2
            width_index = (frames.shape[2] - self.crop_size) // 2

        frames = frames[
            time_index:time_index + self.min_frame_count,
            height_index: height_index + self.crop_size,
            width_index: width_index + self.crop_size, :
        ]

        return frames

    def _load_video(self, video_path):
        capture = cv2.VideoCapture(video_path)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        assert frame_count >= self.min_frame_count
        extract_freq = self.extract_freq
        while frame_count // extract_freq <= self.min_frame_count:
            extract_freq -= 1

        frame_idx = 0
        retaining = True
        frames = []
        while frame_idx < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if frame_idx % extract_freq == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    frames.append(frame)
            frame_idx += 1
        assert len(frames) >= self.min_frame_count
        frames = np.stack(frames)

        # Release the VideoCapture once it is no longer needed
        capture.release()
        return frames
