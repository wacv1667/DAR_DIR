import torch
import random
from torchvision.transforms import functional as F
import pytorchvideo.data
from torch.utils.data import Dataset, DataLoader
from typing import List


class DriverFocusCrop(object):
    """Randomly crop the area where the driver is.
    Gracefully copied this code from: https://github.com/yaorong0921/Driver-Intention-Prediction/blob/master/spatial_transforms.py
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self,
                 train:bool=True,
                 interpolation=F.InterpolationMode.BILINEAR):
        
        self.train = train
        self.interpolation = interpolation

    def __call__(self, video):
        
        # Randomize crop parameters  
        self.randomize_parameters()
        # Get video dimensions
        _, _, image_height, image_width = video.shape

        # Apply cropping on each frame in the video
        video = video[:, :, self.tl_y:image_height-self.tl_y1, self.tl_x:image_width-self.tl_x1]
        # Resize back to the desired output size
        video = torch.nn.functional.interpolate(video, size=(224, 224), mode="bilinear")

        return video

    def randomize_parameters(self):

        self.tl_x = random.randint(200, 400)
        self.tl_y = random.randint(0, 100)
        self.tl_x1 = random.randint(200, 400)
        self.tl_y1 = random.randint(0, 100)


class SequentialRandomSampler:
    def __init__(self, n=4, segments=4):
        self.segments = segments
        self.n = n
    
    def sample_indices(self, num_frames:int) -> List: 
        segment_size = num_frames // self.segments
        segments = []
        
        for i in range(self.segments):
            start = i * segment_size
            end = start + segment_size
            segment = list(range(start,end))
            if segment:
                random_index = random.randint(0, len(segment) - self.n)
                selected_items = segment[random_index:random_index + self.n]
                segments.extend(selected_items)
        return segments
    
    def __call__(self, video:torch.Tensor):
        num_frames = video.shape[1]
        samples = self.sample_indices(num_frames)
        return video[:,samples,...]
