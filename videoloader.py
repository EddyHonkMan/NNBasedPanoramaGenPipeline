import math
import numpy as np
import cv2  # if cv2 code completion is not working, check this https://youtrack.jetbrains.com/issue/PY-54649
from tqdm import tqdm
import json

def show_video(frames):
    if frames:
        for frame in frames:
            cv2.imshow('Frame', frame)

            # 20 is in milliseconds, try to increase the value, say 50 and observe
            cv2.waitKey(20)

        cv2.destroyAllWindows()
    else:
        print("No frames stored")


class VideoLoader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.fps = 0
        self.frame_count = 0
        self.frames = self.load_video_frames(file_name)
        self.motion_vectors_between_all_consecutive_frames = []
        self.choice = []
        self.foregrounds = []
        self.frame_height = None
        self.frame_width = None
        self.YUV_frames = None

    def load_video_frames(self, file_path):
        frames = []
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error opening the video file")
        else:
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("Frame Rate : ", self.fps, "frames per second")
            print("Frame count : ", int(self.frame_count))
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break

        cap.release()
        return frames

    def generate_motion_vector(self, prev, cur, block_size, search_area_size):
        # prev is the frame before, cur is current frame
        res = np.ndarray(shape=(self.height // block_size, self.width // block_size, 2), dtype=int)
        for x in range(0, self.width, block_size):
            for y in range(0, self.height, block_size):
                # we choose the up left most pixel in one block
                minimum_MAD = math.inf
                vector = (0, 0)
                for i in range(max(0, x - search_area_size), min(x + search_area_size, self.width - block_size) + 1):
                    for j in range(max(0, y - search_area_size), min(y + search_area_size, self.height - block_size) + 1):
                        tmp = 0
                        for k in range(block_size):
                            for l in range(block_size):
                                tmp += abs(int(cur[y + l][x + k][0]) - int(prev[j + l][i + k][0]))
                        tmp /= (block_size * block_size)
                        if tmp < minimum_MAD:
                            vector = (i - x, j - y)
                            minimum_MAD = tmp
                res[y // block_size][x // block_size] = vector
        return res



    def foreground_background_separate(self, block_size, search_area_size):
        # check if block_size is compatible with frame size
        (self.height, self.width, _) = self.frames[0].shape
        if self.height % block_size != 0 or self.width % block_size != 0:
            print("Error: block_size is not compatible with frame size!")
            return

        # first convert the frames from RGB to YUV color space
        self.YUV_frames = [cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2YUV) for i in range(len(self.frames))]

        print('Generate motion vector...')
        # generate motion vector between consecutive frames, using given block_size and search_area_size
        n = len(self.YUV_frames)
        for i in tqdm(range(n - 1)):
            self.motion_vectors_between_all_consecutive_frames.append(self.generate_motion_vector(self.YUV_frames[i], self.YUV_frames[i + 1], block_size, search_area_size))
        np.save('{}.npy'.format(self.file_name), self.motion_vectors_between_all_consecutive_frames)
        # separate background and foreground for each frame starting from the second frame
        print('Separate background and foreground...')
        # for f in self.motion_vectors_between_all_consecutive_frames:
        #     mean = np.mean(f)
        #     mean_diff = np.mean([np.abs(mv, mean) for mv in f])
        #     tmp = np.ndarray(shape=(f.shape[:2]), dtype=bool)
        #     tmp.fill(False)
        #     for i in range(len(f)):
        #         for j in range(len(f[i])):
        #             if abs(f[i][j] - mean) > mean_diff:
        #                 tmp[i][j] = True
        #     self.choice.append(tmp)
        motion_vectors_in_all_frames = []
        for f in self.motion_vectors_between_all_consecutive_frames:
            # currrent frame average macroblock motion vector
            mean = np.mean(np.mean(f, axis=1), axis=0)
            # NEED CHECK!!!!!!!
            mean_diff = np.mean([np.abs(mv - mean) for i in range(len(f)) for mv in f[i]])
            sum_motion_vector = np.array([0.0, 0.0])
            for blocks in f:
                for block in blocks:
                    sum_motion_vector += np.abs(block - mean)
            cur_motion_vector = sum_motion_vector / [len(f) * len(f[0]), len(f) * len(f[0])]

            motion_vectors_in_all_frames.append(cur_motion_vector)
            tmp = np.ndarray(shape=(f.shape[:2]), dtype=bool)
            tmp.fill(False)
            for i in range(len(f)):
                for j in range(len(f[i])):

                    if np.abs(f[i][j] - mean).any() > mean_diff.any():
                        tmp[i][j] = True
            self.choice.append(tmp)

        # fill background and foreground track
        print('File background and foreground...')
        for i in range(1, len(self.frames)):
            foreground = np.ndarray(shape=self.frames[0].shape, dtype=np.uint8)
            foreground.fill(255)
            for x in range(len(self.choice[i - 1])):
                for y in range(len(self.choice[i - 1][x])):
                    if self.choice[i - 1][x][y]:
                        for k in range(block_size * x, block_size * (x + 1)):
                            for l in range(block_size * y, block_size * (y + 1)):
                                foreground[k][l], self.frames[i][k][l] = self.frames[i][k][l], foreground[k][l]
            self.foregrounds.append(foreground)

        # now the foreground frames are in self.foregrounds, background frames are still in self.frames





