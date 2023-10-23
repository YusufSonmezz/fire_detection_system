import os
import cv2
from dataclasses import dataclass
import tqdm


@dataclass
class VideoConfig:
    video_folder_path: str = os.path.join('data/videos', '.')
    raw_frame_path: str = os.path.join('data/raw_img')

class Video2Frame:
    def __init__(self):
        self.video_config = VideoConfig()
    
    def seperate_frames_from_videos(self):
        '''
        Read vidoes frame by frame. Save specific frames to image folder.
        '''
        video_path_names = os.listdir(self.video_config.video_folder_path)

        for video_name in tqdm.tqdm(video_path_names):
            video_path = os.path.join(self.video_config.video_folder_path, video_name)
            
            video = cv2.VideoCapture(video_path)

            # Getting fps helps to have different images. If we get all frames from video,
            # then we have lots of same images. Same images causes unnecessary multitude.
            fps = int(video.get(cv2.CAP_PROP_FPS))

            frame_count = 0

            while True:
                # ret: boolean, if frame is read succesfully.
                ret, frame = video.read()
                
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % int(fps / 3) == 0:
                    frame_filename = f'frame_{video_name.replace(".mp4", "")}_{frame_count:04d}.jpg'
                    cv2.imwrite(os.path.join(self.video_config.raw_frame_path, frame_filename), frame)

        video.release()

        


if __name__ == "__main__":
    video_preparation = Video2Frame()
    video_preparation.seperate_frames_from_videos()