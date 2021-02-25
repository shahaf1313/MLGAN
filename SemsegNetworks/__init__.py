from SemsegNetworks.deeplab import Deeplab
from SemsegNetworks.fcn8s import VGG16_FCN8s
import torch.optim as optim
from constants import NUM_CLASSES


def CreateSemsegModel(args):
    model, optimizer = None, None
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=NUM_CLASSES)
        optimizer = optim.SGD(model.optim_parameters(args),
                              lr=args.lr_semseg, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

    if args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=NUM_CLASSES)
        optimizer = optim.Adam(
        [
            {'params': model.get_parameters(bias=False)},
            {'params': model.get_parameters(bias=True),
             'lr': args.lr_semseg * 2}
        ],
        lr=args.lr_semseg,
        betas=(0.9, 0.99))
        optimizer.zero_grad()
    model.to(args.device)
    return model, optimizer


