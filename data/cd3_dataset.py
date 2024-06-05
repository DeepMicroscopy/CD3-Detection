from object_detection_fastai.helper.fastai_helpers import pil2tensor
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2

class CD3Dataset(Dataset):
    def __init__(self, slide, level, patch_size, mean = torch.FloatTensor([0.7324, 0.7587, 0.7719]), std = torch.FloatTensor([0.147 , 0.132 , 0.1246])):
        self.slide = slide
        self.level = level
        self.down_factor = self.slide.level_downsamples[level] 
        self.patch_size = patch_size
        self.coordlist = self.__get_coordlist__()
        self.mean = mean
        self.std = std

    def __get_coordlist__(self, overlap=0.5):
        # Preprocess WSI
        downsamples_int = [int(x) for x in self.slide.level_downsamples]
        ds = 32 if 32 in downsamples_int else 16
        notWSI = False if (32 in downsamples_int or 16 in downsamples_int) else True

        # if not a WSI, all tiles are calculated
        if notWSI:
            activeMap = np.ones((int(self.slide.dimensions[1]/ds),int(self.slide.dimensions[0]/ds)))
            overview=np.ones((int(self.slide.dimensions[1]/ds),int(self.slide.dimensions[0]/ds),3))
        # else, use Otsu thresholding to detect foreground
        else:
            ds_level = np.where(np.abs(np.array(self.slide.level_downsamples)-ds)<1)[0][0]
            overview = self.slide.read_region(level=ds_level, location=(0,0), size=self.slide.level_dimensions[ds_level])
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(overview)[:,:,0:3],cv2.COLOR_BGR2GRAY)
            # OTSU thresholding
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # dilate
            dil = cv2.dilate(thresh, kernel = np.ones((7,7),np.uint8))
            # erode
            activeMap = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))

        x_steps = range(0, int(self.slide.level_dimensions[0][0]),
                    int(self.patch_size * self.down_factor * overlap))
        y_steps = range(0, int(self.slide.level_dimensions[0][1]),
                    int(self.patch_size * self.down_factor * overlap))
        
        coordlist = []
        step_ds = int(np.ceil(float(self.patch_size*self.down_factor)/ds))
        for y in y_steps:
            for x in x_steps:
                x_ds = int(np.floor(float(x)/ds))
                y_ds = int(np.floor(float(y)/ds))
                needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.9*step_ds*step_ds
                if (needCalculation):
                    coordlist.append([x,y])
        return coordlist

    def __len__(self):
        return len(self.coordlist)

    def __getitem__(self, idx):
        x,y = self.coordlist[idx]
        patch = np.array(self.slide.read_region(location=(int(x), int(y)),level=self.level, size=(self.patch_size, self.patch_size)))[:, :, :3]
        patch = pil2tensor(patch / 255., np.float32)
        patch = transforms.Normalize(self.mean, self.std)(patch)
        return patch, x, y    