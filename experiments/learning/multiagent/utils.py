import wandb
import os
import subprocess
from datetime import datetime


def video_recordings_to_wandb_table(videos_folder, fr=15):
    video_table = wandb.Table(columns=["date", "video"])
    for recording_folder in os.listdir(videos_folder):
        recording_folder_path = os.path.join(videos_folder, recording_folder)
        if len(os.listdir(recording_folder_path)) > 0:
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


def configure_and_create_folders(output_folder, ARGS):
    exp = ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo + \
        '-'+ARGS.obs.value+'-'+ARGS.act.value

    filename = os.path.dirname(os.path.abspath(
        __file__))+f'/{output_folder}/results-'+exp+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    videos_folder = os.path.dirname(os.path.abspath(
        __file__))+f'/{output_folder}/videos-'+exp+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder+'/')

    eval_folder = os.path.dirname(os.path.abspath(
        __file__))+f'/{output_folder}/eval-'+exp+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder+'/')

    eval_videos = f"{eval_folder}/videos"
    if not os.path.exists(eval_videos):
        os.makedirs(eval_videos+'/')

    eval_logs = f"{eval_folder}/logs"
    if not os.path.exists(eval_logs):
        os.makedirs(eval_logs+'/')

    return exp, filename, videos_folder, eval_folder, eval_videos, eval_logs

 #### Print out current git commit hash #####################
    # if platform == "linux" or platform == "darwin":
    #     git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    #     with open(filename+'/git_commit.txt', 'w+') as f:
    #         f.write(str(git_commit))
