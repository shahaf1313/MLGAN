from SinGAN.manipulate import *
from SinGAN.training import *
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
    source_loader_iter, target_loader_iter = iter(source_loader), iter(target_loader)

    opt.era_size = np.maximum(len(target_loader.dataset), len(source_loader.dataset))
    opt.steps_per_era = int(np.floor(opt.era_size / opt.batch_size))
    source_loader.dataset.SetEraSize(opt.era_size)
    target_loader.dataset.SetEraSize(opt.era_size)

    opt.source_loader = source_loader
    opt.target_loader = target_loader

    src_img, src_lbl, src_shapes, src_names = source_loader_iter.next()  # new batch source
    trg_img, trg_lbl, trg_shapes, trg_names = target_loader_iter.next()  # new batch target

    del source_loader_iter
    del target_loader_iter

    Gs = []
    Zs = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    real = functions.read_image(opt)
    functions.adjust_scales2image(src_img, opt)
    train(opt, Gs, Zs, NoiseAmp, src_img, trg_img)
    # SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
