import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import datetime
import math
from PIL import Image
from torchvision import transforms
from skimage import io as img
from skimage import color, morphology, filters
from SinGAN.imresize import imresize
import os
import sys
import random
from sklearn.cluster import KMeans

def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img, opt.ref_image))
    return np2torch(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


# def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

# def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp, 0, 1)
    return inp


def save_image(real_cpu, receptive_feild, ncs, epoch_num, file_name):
    fig, ax = plt.subplots(1)
    if ncs == 1:
        ax.imshow(real_cpu.view(real_cpu.size(2), real_cpu.size(3)), cmap='gray')
    else:
        # ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0, 0), receptive_feild, receptive_feild, linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)


def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


def plot_learning_curves(G_loss, D_loss, epochs, label1, label2, name):
    fig, ax = plt.subplots(1)
    n = np.arange(0, epochs)
    plt.plot(n, G_loss, n, D_loss)
    # plt.title('loss')
    # plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1, label2], loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)


def plot_learning_curve(loss, epochs, name):
    fig, ax = plt.subplots(1)
    n = np.arange(0, epochs)
    plt.plot(n, loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)


def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def imresize_torch(image_batch, scale, opt):
    new_size = np.ceil(scale * np.array([image_batch.shape[2], image_batch.shape[3]])).astype(np.int)
    return nn.functional.interpolate(image_batch, size=(new_size[0], new_size[1]), mode='bicubic')
    # to_pil = transforms.ToPILImage()
    # to_torch = transforms.PILToTensor()
    # resized_batch = None
    # for k in range(image_batch.shape[0]):
    #     pil_im = to_pil(image_batch[k])
    #     new_size = (np.ceil(scale * np.array(pil_im.size))).astype(np.int)
    #     new_image = pil_im.resize(new_size, Image.BICUBIC)
    #     new_image = to_torch(new_image)
    #     if resized_batch is None:
    #         resized_batch = new_image.unsqueeze(0)
    #     else:
    #         resized_batch = torch.cat((resized_batch, new_image.unsqueeze(0)), 0)
    # return resized_batch.to(opt.device)

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    # print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),  # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def read_image_dir(dir, opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
    # x = x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = x[:, :, 0:3]
    return x


def save_networks(netDst, netGst, netDts, netGts, opt):
    torch.save(netDst.state_dict(), '%s/netDst.pth' % (opt.outf))
    torch.save(netGst.state_dict(), '%s/netGst.pth' % (opt.outf))
    torch.save(netDts.state_dict(), '%s/netDts.pth' % (opt.outf))
    torch.save(netGts.state_dict(), '%s/netGts.pth' % (opt.outf))


def adjust_scales2image(H, W, opt):
    opt.max_size = max(H, W)
    opt.min_size = math.ceil(min(H, W) * math.pow(opt.scale_factor, opt.num_scales))
    opt.stop_scale = opt.num_scales


def adjust_scales2image_SR(real_, opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    # opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real


def creat_reals_pyramid(real, opt):
    reals = []
    real = real[:, 0:3, :, :]
    for i in range(0, opt.curr_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    return reals


# def load_trained_pyramid(opt, mode_='train'):
#     # dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
#     mode = opt.mode
#     opt.mode = 'train'
#     if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
#         opt.mode = mode
#     dir = generate_dir2save(opt)
#     if (os.path.exists(dir)):
#         Gs = torch.load('%s/Gs.pth' % dir)
#         Zs = torch.load('%s/Zs.pth' % dir)
#         reals = torch.load('%s/reals.pth' % dir)
#         NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
#     else:
#         print('no appropriate trained model is exist, please train first')
#     opt.mode = mode
#     return Gs, Zs, reals, NoiseAmp


def generate_in2coarsest(reals, scale_v, scale_h, opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else:  # if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.out_ = 'TrainedModels/%s' % datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S')
    try:
        os.makedirs(opt.out_)
    except OSError:
        pass
    opt.logger = Logger(os.path.join(opt.out_, 'log.txt'))
    sys.stdout = opt.logger
    opt.nfc_init = opt.nfc
    opt.min_nfc = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)[1:-1].strip(' ')
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def calc_init_scale(opt):
    in_scale = math.pow(1 / 2, 1 / 3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale, iter_num


def quant(prev, device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    # x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x, centers


def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    # x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask, opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask, opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0, vmax=1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask
