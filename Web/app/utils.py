from drone.circle import DroneController
from flask_socketio import SocketIO
import time
import os
import cv2
from app.dl.dl import DLController

class Video2Frame:

    def split_video_into_pieces(self, video_path:str, num_points:int):
        duration = self.get_video_length(video_path)

        portition_size = duration // num_points

        portitions = []

        for i in range(num_points):
            start_frame = i * portition_size
            end_frame = (i + 1) * portition_size

            if i == num_points - 1:
                end_frame = duration
            
            portitions.append([start_frame, end_frame])
        
        return portitions

    
    def get_video_length(self, video_path:str):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        # Get the frames per second (fps) and total number of frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the duration of the video in seconds
        duration_seconds = total_frames / fps

        # Release the video capture object
        cap.release()

        return duration_seconds


    def take_frames_from_certain_interval(self, frame_per: int, video_path: str, start_time: float, end_time: float):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []

        frame_count = 0
        while (cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame) & (frame_count % (fps // frame_per) == 0):
            frame_count += 1

            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        return frames



        


def send_message_to_client(socketio, message):
    socketio.emit('update_message', {'message': message})

def start_drone(socketio, diameter:float, altitude:int, num_points:int):
    drone = DroneController(diameter, altitude, num_points)
    v2f = Video2Frame()
    dl_model_path = "app/static/dl_model/best_model.pth"
    dl = DLController(dl_model_path)

    # Guided mode
    drone.mode_guided()
    send_message_to_client(socketio, "Drone mode is configured to GUIDED.")

    drone.takeoff()
    send_message_to_client(socketio, "Drone is taking off right now.")

    send_message_to_client(socketio, f"Drone altitude is {drone.get_current_altitude()} metres.")
    time.sleep(3)
    send_message_to_client(socketio, f"Drone altitude is {drone.get_current_altitude()} metres.")

    ## Video preparing
    video_path = "app/static/data/video/fire_nofire.mp4"

    portitions = v2f.split_video_into_pieces(video_path, num_points)

    for idx, waypoint in enumerate(range(num_points)):
        drone.go_to(waypoint)
        frames = v2f.take_frames_from_certain_interval(5, video_path, int(portitions[waypoint][0]), int(portitions[waypoint][1]))

        for frame in frames:
            output, prob = dl.predict_image(frame)
            
            if (output == 1) & (prob >= 0.90):
                dl.save_image(frame, output, prob, "app/static/data/predicted", idx)

                drone.mode_rtl()

                return frame, output, prob, drone.get_current_location()
    drone.mode_rtl()


if __name__ == "__main__":
    v2f = Video2Frame()
    video_path = "app/static/data/video/fire_nofire.mp4"
    portitions = v2f.split_video_into_pieces(video_path, 36)
    print(portitions)
    duration = v2f.get_video_length(video_path)
    print("Duration is ..: ", duration)

    dl_model_path = "app/static/dl_model/best_model.pth"
    dl = DLController(dl_model_path)

    for idx, waypoint in enumerate(range(36)):
        frames = v2f.take_frames_from_certain_interval(5, video_path, int(portitions[waypoint][0]), int(portitions[waypoint][1]))

        for frame in frames:
            output, prob = dl.predict_image(frame)
            
            if (output == 1) & (prob >= 0.90):
                print("output is ..: ", output, "prob is ..: ", prob)
                print(int(portitions[waypoint][0]), int(portitions[waypoint][1]))
                dl.save_image(frame, output, prob, "app/static/data/predicted", idx)
    

