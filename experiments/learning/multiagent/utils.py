import wandb
import os
import subprocess
from datetime import datetime

def video_recordings_to_wandb_table(videos_folder, fr=15):
    video_table = wandb.Table(columns=["date", "video"])
    for recording_folder in os.listdir(videos_folder):
        recording_folder_path = os.path.join(videos_folder, recording_folder)
        if len(os.listdir(recording_folder_path))>0:
            recording_gif_path = f"{recording_folder_path}/output.gif"
            _ = subprocess.call([
                "ffmpeg", 
                "-framerate", str(fr),
                "-start_number", "0", 
                "-i", f"{recording_folder_path}/frame_%d.png", 
                recording_gif_path
            ])
            substring = "_".join(recording_folder.split("_")[-2:])
            loaded_date = datetime.strptime(substring, "%m.%d.%Y_%H.%M.%S")
            print(loaded_date)
            loaded_gif = wandb.Video(recording_gif_path, None, fr)
            video_table.add_data(loaded_date, loaded_gif)
    return video_table