import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--out', help='output folder', default='Output')

    # Dataset parameters:
    parser.add_argument("--source", type=str, default='gta5', help="source dataset : gta5 or synthia")
    parser.add_argument("--target", type=str, default='cityscapes', help="target dataset : cityscapes")
    parser.add_argument("--src_data_dir", type=str, default='/home/shahaf/data/GTA5', help="Path to the directory containing the source dataset.")
    parser.add_argument("--src_data_list", type=str, default='./dataset/gta5_list/train.txt', help="Path to the listing of images in the source dataset.")
    parser.add_argument("--trg_data_dir", type=str, default='/home/shahaf/data/cityscapes', help="Path to the directory containing the target dataset.")
    parser.add_argument("--trg_data_list", type=str, default='./dataset/cityscapes_list/train.txt', help="List of images in the target dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for each worker")

    # networks hyper parameters:
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nfc', type=int, default=16)
    parser.add_argument('--min_nfc', type=int, default=16)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)  # pow(0.5,1/6))
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=None)
    parser.add_argument('--max_size', type=int, help='image maximal size at the largest scale', default=None)
    parser.add_argument('--num_scales', type=int, help='number of scales in the pyramid', default=None)


    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)

    # Miscellaneous parameters:
    parser.add_argument("--tb_logs_dir", type=str, required=False, default='./runs', help="Path to Tensorboard logs dir.")
    parser.add_argument("--checkpoints_dir", type=str, required=False, default='./checkpoints', help="Where to save snapshots of the model.")

    return parser
