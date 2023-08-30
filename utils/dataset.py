import torch
import random
import torch.nn.functional as F
import pytorchvideo.data
from typing import List
from torch.utils.data import Dataset, DataLoader
from glob import glob
import  pandas as pd

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from .transforms import SequentialRandomSampler, DriverFocusCrop


class train_dataset(Dataset):
    def __init__(self,
                 base_dir:str="brain4cars_data/",
                 fold_num:int=0,
                 resize_to = (224,224),
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225], 
                 inside:bool=True,
                 outside:bool=True):
        self.dataset_list = [path for path in glob(f"{base_dir}/train/fold{str(fold_num)}/**/*.avi", recursive=True) if path.split("/")[-2] != "road_vids"]
        self.class_labels = sorted({str(path).split("/")[-2] for path in self.dataset_list})
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_classes = len(self.label2id)
        self.transform_inside = Compose([ApplyTransformToKey(
                                key="video",
                                transform=Compose([
                                        SequentialRandomSampler(),
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        DriverFocusCrop(),
                                        RandomShortSideScale(min_size=256, max_size=320),
                                        RandomCrop(resize_to),
                                    ]),),])
        self.transform_outside = Compose([ApplyTransformToKey(
                                key="video",
                                transform=Compose([
                                        SequentialRandomSampler(),
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        RandomShortSideScale(min_size=256, max_size=320),
                                        RandomCrop(resize_to),
                                    ]),),])
        self.flip_transform = RandomHorizontalFlip(p=1.0)
        self.inside = inside
        self.outside = outside
        
    
    def __len__(self):
        return len(self.dataset_list)
    
    # def infer_context_vector(self, ego_lane, num_lanes, intersect):
    #     #{0: 'end_action', 1: 'lchange', 2: 'lturn', 3: 'rchange', 4: 'rturn'}
    #     context = [0,0,0,0,0]

    #     # if there is only one lane it is not possible to perform a lane change
    #     if num_lanes == 1:
    #         context[1] = 1
    #         context[3] = 1

    #     # if you are it the most left lane, then is not possible to perform a left lane change
    #     if ego_lane == num_lanes:
    #         context[1] = 1
    #     # if you are in the most right lane, then is not possible to perform a right lane change
    #     if ego_lane == 1:
    #         context[3] = 1  

    #     # if there are lanes available to your left, then it is possible to perform a left lane change
    #     if ego_lane < num_lanes:
    #         context[1] = 0

    #     # if you are in the most left lane, near an intersection, and there is more than 1 lane
    #     # then a right turn is probably not allowed
    #     if (intersect == 1) and (ego_lane == num_lanes) and (num_lanes > 1):
    #         context[4] = 1
    #     # if you are in the most right lane, near an intersection, and there is more than 1 lane
    #     # then a left turn is probably not allowed
    #     if (intersect == 1) and (ego_lane == 1) and (num_lanes > 1):
    #         context[2] = 1
    #     return torch.tensor(context)
    
    def load_context_vector(self,file_path):
        df = pd.read_csv(file_path, header=None) 
        ego_lane, num_lanes, intersect = df.values[0].tolist()
        context_vector = self.infer_context_vector(ego_lane, num_lanes, intersect)
        return context_vector
    
    def load_outside_video_path(self, video_path):
        label = str(video_path).split("/")[-2]
        return video_path.replace(label, "road_vids").replace("video_", "")
    
    def load_file_paths(self, index):
        inside_video_path = self.dataset_list[index]
        outside_video_path = self.load_outside_video_path(inside_video_path)
        context_path = outside_video_path.replace(".avi", ".txt")
        return inside_video_path, outside_video_path, context_path
    
    def load_video(self, video_path, inside=True):
        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
         # Load the desired clip
        fps = video.duration.denominator
        frames = fps * video.duration.numerator 
        end_sec = frames/fps
        video= video.get_clip(start_sec=0, end_sec=end_sec)
        video = self.transform_inside(video) if inside else self.transform_outside(video)
        return video["video"]
    
    def load_target(self, video_path):
        label =  self.label2id[str(video_path).split("/")[-2]]
        one_hot_label = F.one_hot(torch.tensor(label), self.num_classes).float()
        return one_hot_label
    
    def flip_video(self, video):
        return torch.stack([self.flip_transform(frame) for frame in video]) 
    
    def label_flip(self, label_tensor):
        """Flip labels and context penalty vectors"""
        driving_straight = label_tensor[0]
        left_lane_change = label_tensor[1]
        left_turn = label_tensor[2]
        right_lane_change = label_tensor[3]
        right_turn = label_tensor[4]
        return torch.tensor([driving_straight, right_lane_change, right_turn, left_lane_change, left_turn])
    
    def horizontal_flip(self, data_dict):
        if random.random() > 0.5:
            try: 
                data_dict["inside"] = self.flip_video(data_dict["inside"]) 
            except KeyError:
                pass
            
            try: 
                data_dict["outside"] = self.flip_video(data_dict["outside"]) 
            except KeyError:
                pass
            data_dict["label"] = self.label_flip(data_dict["label"])
            # data_dict["context"] = self.label_flip(data_dict["context"])
            
        return data_dict
            
    def __getitem__(self, index):
        inside_video_path, outside_video_path, context_path = self.load_file_paths(index)
        target = self.load_target(inside_video_path)
        
        context_vector = self.load_context_vector(context_path)
        data_dict =  {"label":target,
                      "context":context_vector}
        
        if self.inside:
            inside_video = self.load_video(inside_video_path, inside=True)
            data_dict["inside"] = inside_video
        if self.outside:
            outside_video = self.load_video(outside_video_path, inside=False)
            data_dict["outside"] = outside_video
        
        data_dict = self.horizontal_flip(data_dict)
        return data_dict


class val_dataset(Dataset):
    def __init__(self,
                 base_dir:str="brain4cars_data/",
                 fold_num:int=0,
                 resize_to = (224,224),
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 inside:bool=True,
                 outside:bool=True):
        self.dataset_list = [path for path in glob(f"{base_dir}/val/fold{str(fold_num)}/**/*.avi", recursive=True) if path.split("/")[-2] != "road_vids"]
        self.class_labels = sorted({str(path).split("/")[-2] for path in self.dataset_list})
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_classes = len(self.label2id)
        self.transform = Compose([ApplyTransformToKey(
                                key="video",
                                transform=Compose([
                                        UniformTemporalSubsample(16),
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        Resize(resize_to),
                                    ]),),])
        self.inside = inside
        self.outside = outside
     
    def __len__(self):
        return len(self.dataset_list)
    
    def load_outside_video_path(self, video_path):
        label = str(video_path).split("/")[-2]
        return video_path.replace(label, "road_vids").replace("video_", "")
    
    def load_file_paths(self, index):
        inside_video_path = self.dataset_list[index]
        outside_video_path = self.load_outside_video_path(inside_video_path)
        return inside_video_path, outside_video_path
    
    def load_video(self, video_path):
        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
         # Load the desired clip
        fps = video.duration.denominator
        frames = fps * video.duration.numerator 
        end_sec = frames/fps
        video= video.get_clip(start_sec=0, end_sec=end_sec)
        video = self.transform(video) 
        return video["video"]
    
    def load_target(self, video_path):
        label =  self.label2id[str(video_path).split("/")[-2]]
        one_hot_label = F.one_hot(torch.tensor(label), self.num_classes).float()
        return one_hot_label
    
    def __getitem__(self, index):
        inside_video_path, outside_video_path = self.load_file_paths(index)
        target = self.load_target(inside_video_path)
        data_dict =  {"label":target}
        
        if self.inside:
            inside_video = self.load_video(inside_video_path)
            data_dict["inside"] = inside_video
        if self.outside:
            outside_video = self.load_video(outside_video_path)  
            data_dict["outside"] = outside_video
        return data_dict
