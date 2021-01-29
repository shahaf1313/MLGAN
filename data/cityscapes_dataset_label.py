import numpy as np
from constants import IGNORE_LABEL
import os.path as osp
from PIL import Image
from torch.utils import data

class cityscapesDataSetLabel(data.Dataset):
    def __init__(self, root, list_path, crop_size, mean=None, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.ignore_label = IGNORE_LABEL
        self.set = set

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(   self.root, "leftImg8bit/%s/%s" % (self.set, name)   )).convert('RGB')
        lbname = name.replace("leftImg8bit", "gtFine_labelIds")
        label = Image.open(osp.join(   self.root, "gtFine/%s/%s" % (self.set, lbname)   ))
        assert image.size == label.size

        image = image.resize( self.crop_size, Image.BICUBIC )
        label = label.resize( self.crop_size, Image.NEAREST )
        image = np.asarray( image, np.float32 )
        label = np.asarray( label, np.float32 )

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image -= self.mean
        image = image.transpose((2, 0, 1))
        image = (image - 128.) / 128  # change from 0..255 to -1..1
        return image.copy(), label_copy.copy(), np.array(size), name

    def SetEraSize(self, era_size):
        if (era_size > len(self.img_ids)):
            self.img_ids = self.img_ids * int(np.ceil(float(era_size) / len(self.img_ids)))
        self.img_ids = self.img_ids[:era_size]

