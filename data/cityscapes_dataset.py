import numpy as np
from data.domainAdaptationDataset import domainAdaptationDataSet
import os.path as osp
from PIL import Image

class cityscapesDataSet(domainAdaptationDataSet):
    def __init__(self, root, list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False):
        super(cityscapesDataSet, self).__init__( root, list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(   self.root, "leftImg8bit/%s/%s" % (self.set, name)   )).convert('RGB')
        image = image.resize(self.crop_size, Image.BICUBIC)

        scales_pyramid, label, label_copy = None, None, None
        if self.get_image_label:
            lbname = name.replace("leftImg8bit", "gtFine_labelIds")
            label = Image.open(osp.join(   self.root, "gtFine/%s/%s" % (self.set, lbname)   ))
            assert image.size == label.size
            label = label.resize( self.crop_size, Image.NEAREST )
            label = np.asarray( label, np.float32 )
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            label_copy = label_copy.copy()

        if self.set == 'train':
            scales_pyramid = self.GeneratePyramid(image)

        if self.set == 'train' and self.get_image_label:
            return scales_pyramid, label_copy
        elif self.set == 'train' and not self.get_image_label:
            return scales_pyramid
        elif self.set is not 'train' and self.get_image_label:
            return image.copy(), label_copy
        elif self.set is not 'train' and not self.get_image_label:
            return image.copy()




