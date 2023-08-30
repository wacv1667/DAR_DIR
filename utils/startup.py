import pandas as pd
from glob import glob 
import os 
import pathlib
import pytorchvideo
import pytorchvideo.data
import warnings
warnings.filterwarnings("ignore")


def road_file_check(path):
    path = path.replace("face_camera", "road_camera").replace("video_","")
    path = path.split("/")
    path.pop(-2)
    if os.path.exists("/".join(path)) is True:
        return True
    else:
        return False

def load_video_paths(base_dir,
                fold_num:int=0,
                min_frames:int=125):
    dataset_list = [x for x in glob(f"{base_dir}/face_camera/**/*.avi", recursive=True)] 
    dataset_list = [x for x in dataset_list if road_file_check(x) is True]
    
    df= pd.read_csv(f"{base_dir}/Driver-Intention-Prediction/datasets/annotation/fold{str(fold_num)}.csv", header=None)
    df_train = df[(df[2] >min_frames) & (df[3] == "training")]    
    df_val = df[(df[2] >min_frames) & (df[3] == "validation")]
    df_val.columns = ["id", "original_label", "num_frames", "dataset_type"]
    df_train.columns = ["id", "original_label", "num_frames", "dataset_type"]
    
    df_train["face_video"] = df_train["id"].apply(lambda x: glob(f"{base_dir}/face_camera/{x}*.avi")[0])
    df_train["road_video"] = df_train["id"].apply(lambda x: glob(f"{base_dir}/road_camera/{x[:-1]}*.avi")[0])
    
    df_val["face_video"] = df_val["id"].apply(lambda x: glob(f"{base_dir}/face_camera/{x}*.avi")[0])
    df_val["road_video"] = df_val["id"].apply(lambda x: glob(f"{base_dir}/road_camera/{x[:-1]}*.avi")[0])
    return df_train, df_val

def symlink_files(fold_num, label, datatype="train"):
    base_dir =  f"./brain4cars_data/{datatype}/fold{fold_num}/"
    files = pd.read_csv(f"{base_dir}{label}.csv")["face_video"].to_list()
    for file in files:
        try:
            file_name = file.split("/")[-1]
            os.symlink(file, f"{base_dir}{label}/{file_name}")
        except FileExistsError:
            pass

def symlink_road_videos(base_dir:str="/home/jovyan/b4c/"):
    for data_type in ["train", "val"]:
        for fold_num in range(5):
            try:
                os.makedirs(f"{base_dir}/brain4cars_data/{data_type}/fold{fold_num}/road_vids")
            except FileExistsError:
                pass
            videos = [x.split('/')[-1].split("video_")[-1].split(".avi")[0] for x in
                              glob(f"{base_dir}/brain4cars_data/{data_type}/fold{fold_num}/**/*.avi", recursive=True)]

            symlink_files = [x for x in glob(f"{base_dir}/road_camera/**/*", recursive=True) 
                                         if x.split('/')[-1].split(".")[0] in videos]

            for file in symlink_files:
                try:
                    file_name = file.split("/")[-1]
                    os.symlink(file, f"{base_dir}/brain4cars_data/{data_type}/fold{fold_num}/road_vids/{file_name}")
                except FileExistsError:
                    pass


def count_frames(video_path):
    video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
    fps = video.duration.denominator
    frames = fps * video.duration.numerator 
    num_frames = frames/fps
    return num_frames

def unlink_short_vids(fold_num:int, min_num_frames=16):
    file_paths =  glob(f"./brain4cars_data/val/fold{fold_num}/road_vids/*.avi") + glob(f"./brain4cars_data/train/fold{fold_num}/road_vids/*.avi") 
    df = pd.DataFrame(columns=['id', 'frames'])
    for path in file_paths:
        vid_id = path.split("/")[-1].replace(".avi","")
        frame_nums = count_frames(path)
        df.loc[len(df)] = [vid_id, frame_nums]

    short_video_list = df[df["frames"] < min_num_frames]["id"].tolist()
    file_paths = []
    for file_id in short_video_list:
        file_paths = file_paths + list(set(glob(f'./brain4cars_data/**/**/**/*{file_id}*avi', recursive=True)))
    for file in file_paths:
        os.unlink(file)


def main():
    base_dir = str(pathlib.Path().resolve())
    for fold_num in range(5):
        df_train, df_val = load_video_paths(base_dir=base_dir, fold_num=fold_num)
        for label in df_train.original_label.unique():
            # create a seperate folder for each validation fold and label
            try:
                os.makedirs(f"./brain4cars_data/val/fold{fold_num}/{label}")
            except Exception as e:
                print(e)
                pass
            try:
                os.makedirs(f"./brain4cars_data/train/fold{fold_num}/{label}")
            except Exception as e:
                print(e)
                pass
            
            # save a file with the labels per video ID
            df_train[df_train["original_label"] == label]["face_video"].to_csv(f"./brain4cars_data/train/fold{fold_num}/{label}.csv")
            df_val[df_val["original_label"] == label]["face_video"].to_csv(f"./brain4cars_data/val/fold{fold_num}/{label}.csv")
        
             # create the symbolic link to the new folder structure
            symlink_files(fold_num, label, datatype="train")
            symlink_files(fold_num, label, datatype="val")
    symlink_road_videos(base_dir)
    
    for fold_num in range(5):
        unlink_short_vids(fold_num)

if __name__ == '__main__':
    main()
