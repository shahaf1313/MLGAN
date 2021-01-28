from SinGAN.manipulate import *
from SinGAN.training import *
from constants import H,W
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)


    source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt)
    opt.era_size = np.maximum(len(target_loader.dataset), len(source_loader.dataset))
    opt.max_batch_size = opt.batch_size

    source_loaders, target_loaders =[], []
    for i in range(opt.num_scales+1):
        source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt)
        source_loader.dataset.SetEraSize(opt.era_size)
        target_loader.dataset.SetEraSize(opt.era_size)
        source_loaders.append(source_loader)
        target_loaders.append(target_loader)
        if i%2==0 and i >0:
            opt.batch_size = int(np.maximum(opt.batch_size/2, 1))

    opt.source_loaders = source_loaders
    opt.target_loaders = target_loaders


    Gs = []
    dir2save = functions.generate_dir2save(opt)

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    real = functions.read_image(opt)
    functions.adjust_scales2image(H, W, opt)
    train(opt)
    # SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
