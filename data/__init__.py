from constants import IMG_MEAN, IMG_CROP_SIZE
from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNDataSet


def CreateSrcDataLoader(args, mode=None):
    if args.source == 'gta5' and mode == 'train_semseg':
        source_dataset = GTA5DataSet(args.data_dir, './dataset/gta5_list/train_semseg_net.txt', crop_size=IMG_CROP_SIZE, mean=IMG_MEAN)
    elif args.source == 'gta5' and mode == 'val_semseg':
        source_dataset = GTA5DataSet(args.data_dir, './dataset/gta5_list/val_semseg_net.txt', crop_size=IMG_CROP_SIZE, mean=IMG_MEAN)
    elif args.source == 'gta5' and mode is None:
        source_dataset = GTA5DataSet(args.src_data_dir, args.src_data_list, crop_size=IMG_CROP_SIZE, mean=IMG_MEAN)
    # elif args.source == 'synthia':
    #     source_dataset = SYNDataSet(args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'], resize=image_sizes['synthia'], mean=IMG_MEAN)
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')

    source_dataloader = data.DataLoader(source_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    return source_dataloader


def CreateTrgDataLoader(args, mode='train'):
    if mode == 'train':
        target_dataset = cityscapesDataSet(args.trg_data_dir,
                                                args.trg_data_list,
                                                crop_size=IMG_CROP_SIZE,
                                                mean=IMG_MEAN,
                                                set=mode)
    elif mode == 'val':
        target_dataset = cityscapesDataSet(args.trg_data_dir,
                                                args.trg_data_list,
                                                crop_size=IMG_CROP_SIZE,
                                                mean=IMG_MEAN,
                                                set=mode)
    else:
        raise Exception("Argument set has not entered properly. Options are train or eval.")

    if mode == 'train':
        target_dataloader = data.DataLoader(target_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            pin_memory=True)
    elif mode == 'val':
        target_dataloader = data.DataLoader(target_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)
    else:
        raise Exception("Argument set has not entered properly. Options are train or eval.")

    return target_dataloader
