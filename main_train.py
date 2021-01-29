from SinGAN.manipulate import *
from SinGAN.training import *
from constants import H,W
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)

    opt.batch_size = len(opt.gpus)*2
    source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt)
    opt.era_size = np.maximum(len(target_loader.dataset), len(source_loader.dataset))

    source_loaders, target_loaders =[], []
    for i in range(opt.num_scales+1):
        if i>5:
            opt.batch_size = len(opt.gpus)
        source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt)
        source_loader.dataset.SetEraSize(opt.era_size)
        target_loader.dataset.SetEraSize(opt.era_size)
        source_loaders.append(source_loader)
        target_loaders.append(target_loader)


    opt.source_loaders = source_loaders
    opt.target_loaders = target_loaders

    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    real = functions.read_image(opt)
    functions.adjust_scales2image(H, W, opt)
    train(opt)
    print('Finished Training.')
