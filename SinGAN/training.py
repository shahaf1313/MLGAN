import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import numpy as np
import time
from SinGAN.imresize import imresize
from SinGAN.functions import imresize_torch
import datetime
from torch.utils.tensorboard import SummaryWriter

def train(opt):
    scale_num = 0
    nfc_prev = 0
    Gst, Gts = [], []
    Dst, Dts = [], []
    opt.tb = SummaryWriter('./runs/%sGPU%d/' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0]))

    while scale_num < opt.stop_scale + 1:
        opt.curr_scale = scale_num
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        Dst_curr, Gst_curr = init_models(opt)
        Dts_curr, Gts_curr = init_models(opt)
        # if (nfc_prev == opt.nfc and opt.curr_scale > 1):
        #     Dst_curr.load_state_dict(torch.load('%s/%d/netDst.pth' % (opt.out_, scale_num - 1)))
        #     Gst_curr.load_state_dict(torch.load('%s/%d/netGst.pth' % (opt.out_, scale_num - 1)))
        #     Dts_curr.load_state_dict(torch.load('%s/%d/netDts.pth' % (opt.out_, scale_num - 1)))
        #     Gts_curr.load_state_dict(torch.load('%s/%d/netGts.pth' % (opt.out_, scale_num - 1)))
        if len(opt.gpus) > 1:
            Dst_curr, Gst_curr = nn.DataParallel(Dst_curr, device_ids=opt.gpus), nn.DataParallel(Gst_curr, device_ids=opt.gpus)
            Dts_curr, Gts_curr = nn.DataParallel(Dts_curr, device_ids=opt.gpus), nn.DataParallel(Gts_curr, device_ids=opt.gpus)

        scale_nets = train_single_scale(Dst_curr, Gst_curr, Dts_curr, Gts_curr, Gst, Gts, Dst, Dts, opt)
        for net in scale_nets:
            net = functions.reset_grads(net, False)
            net.eval()
        Dst_curr, Gst_curr, Dts_curr, Gts_curr = scale_nets

        Gst.append(Gst_curr)
        Gts.append(Gts_curr)
        Dst.append(Dst_curr)
        Dts.append(Dts_curr)

        torch.save(Gst, '%s/Gst.pth' % (opt.out_))
        torch.save(Gts, '%s/Gts.pth' % (opt.out_))
        torch.save(Dst, '%s/Dst.pth' % (opt.out_))
        torch.save(Dts, '%s/Dts.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del Dst_curr, Gst_curr, Dts_curr, Gts_curr
    opt.tb.close()
    return

def train_single_scale(netDst, netGst, netDts, netGts, Gst: list, Gts: list, Dst: list, Dts: list, opt):
        # setup optimizers and schedulers:
        optimizerDst = optim.Adam(netDst.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGst = optim.Adam(netGst.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerDts = optim.Adam(netDts.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGts = optim.Adam(netGts.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        schedulerDst = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerDst, milestones=[1600], gamma=opt.gamma)
        schedulerGst = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerGst, milestones=[1600], gamma=opt.gamma)
        schedulerDts = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerDts, milestones=[1600], gamma=opt.gamma)
        schedulerGts = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerGts, milestones=[1600], gamma=opt.gamma)

        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))

        discriminator_steps = 0
        generator_steps = 0
        steps = 0
        print_int = 0
        save_pics_int = 0
        epoch_num = 1
        start = time.time()
        keep_training = True

        while keep_training:
            print('scale %d: starting epoch %d...' % (opt.curr_scale, epoch_num))
            epoch_num += 1
            for batch_num, (source_scales, target_scales) in enumerate(zip(opt.source_loaders[opt.curr_scale], opt.target_loaders[opt.curr_scale])):
                if steps > opt.num_steps:
                    keep_training = False
                    break

                # Move scale tensors to CUDA:
                for i in range(len(source_scales)):
                    source_scales[i] = source_scales[i].to(opt.device)
                    target_scales[i] = target_scales[i].to(opt.device)

                # Create pyramid concatenation:
                prev_sit = concat_pyramid(Gst, source_scales, m_noise, m_image, opt)
                prev_sit = m_image(prev_sit)
                prev_tis = concat_pyramid(Gts, target_scales, m_noise, m_image, opt)
                prev_tis = m_image(prev_tis)

                ############################
                # (1) Update D networks: maximize D(x) + D(G(z))
                ###########################
                images = None
                for j in range(opt.Dsteps):
                    #train discriminator networks between domains (S->T, T->S)

                    #S -> T:
                    D_x, D_G_z, errD = adversarial_disciriminative_train(netDst, optimizerDst, netGst, prev_sit, target_scales[opt.curr_scale], source_scales, m_noise, opt)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorLoss' % opt.curr_scale, errD.item(), discriminator_steps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z, discriminator_steps)


                    # T -> S:
                    D_x, D_G_z, errD = adversarial_disciriminative_train(netDts, optimizerDts, netGts, prev_tis, source_scales[opt.curr_scale], target_scales, m_noise, opt)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorLoss' % opt.curr_scale, errD.item(), discriminator_steps)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z, discriminator_steps)

                    discriminator_steps += 1

                ############################
                # (2) Update G networks: maximize D(G(z)), minimize Gst(Gts(s))-s and vice versa
                ###########################

                for j in range(opt.Gsteps):
                    # train generator networks between domains (S->T, T->S)

                    # S -> T:
                    errG = adversarial_generative_train(netGst, optimizerGst, netDst, prev_sit, source_scales, m_noise, opt)
                    opt.tb.add_scalar('Scale%d/ST/GeneratorAdversarialLoss' % opt.curr_scale, errG.item(), generator_steps)

                    # T -> S:
                    errG = adversarial_generative_train(netGts, optimizerGts, netDts, prev_tis, target_scales, m_noise, opt)
                    opt.tb.add_scalar('Scale%d/TS/GeneratorAdversarialLoss' % opt.curr_scale, errG.item(), generator_steps)

                    # Identity Loss:
                    # (netGx, optimizerGx, x, x_scales, prev_x,
                    #  netGy, optimizerGy, y, y_scales, prev_y,
                    #  m_noise, opt)
                    loss_x, loss_y, loss_id, images = cycle_consistency_loss(netGst, optimizerGst, source_scales[opt.curr_scale], source_scales, prev_sit,
                                                                             netGts, optimizerGts, target_scales[opt.curr_scale], target_scales, prev_tis,
                                                                             m_noise, opt)
                    opt.tb.add_scalar('Scale%d/Identity/LossSTS' % opt.curr_scale, loss_x.item(), generator_steps)
                    opt.tb.add_scalar('Scale%d/Identity/LossTST' % opt.curr_scale, loss_y.item(), generator_steps)
                    opt.tb.add_scalar('Scale%d/Identity/Loss' % opt.curr_scale, loss_id.item(), generator_steps)

                    generator_steps += 1

                if int(steps/opt.print_rate) >= print_int or steps == 0:
                    elapsed = time.time() - start
                    print('scale %d:[%d/%d] ; elapsed time = %.2f secs per step' %
                          (opt.curr_scale, print_int*opt.print_rate, opt.num_steps, elapsed/opt.print_rate))
                    start = time.time()
                    print_int += 1

                if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                    s     = norm_image(source_scales[opt.curr_scale][0])
                    t     = norm_image(target_scales[opt.curr_scale][0])
                    sit   = norm_image(images[0][0])
                    sitis = norm_image(images[1][0])
                    tis   = norm_image(images[2][0])
                    tisit = norm_image(images[3][0])
                    opt.tb.add_image('Scale%d/source' % opt.curr_scale, s, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_in_traget' % opt.curr_scale, sit, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_in_traget_in_source' % opt.curr_scale, sitis, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target' % opt.curr_scale, t, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target_in_source' % opt.curr_scale, tis, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target_in_source_in_target' % opt.curr_scale, tisit, save_pics_int*opt.save_pics_rate)

                    save_pics_int += 1

                steps = np.minimum(generator_steps, discriminator_steps)

                # schedulerDst.step()
                # schedulerGst.step()
                # schedulerDts.step()
                # schedulerGts.step()

        if (len(opt.gpus) > 1):
            functions.save_networks(netDst.module, netGst.module, netDts.module, netGts.module, opt)
            return netDst.module, netGst.module, netDts.module, netGts.module
        else:
            functions.save_networks(netDst, netGst, netDts, netGts, opt)
            return netDst, netGst, netDts, netGts

def adversarial_disciriminative_train(netD, optimizerD, netG, prev, real_images, from_scales, m_noise, opt):
    # train with real image
    netD.zero_grad()
    output = netD(real_images).to(opt.device)
    errD_real = -1 * opt.lambda_adversarial * output.mean()
    errD_real.backward(retain_graph=True)
    D_x = errD_real.item()

    # train with fake
    curr = m_noise(from_scales[opt.curr_scale])
    fake_images = netG(curr, prev)
    output = netD(fake_images.detach())
    errD_fake =  opt.lambda_adversarial * output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = errD_fake.item()

    gradient_penalty =  opt.lambda_adversarial * functions.calc_gradient_penalty(netD, real_images, fake_images, opt.lambda_grad, opt.device)
    gradient_penalty.backward()

    errD = errD_real + errD_fake + gradient_penalty
    optimizerD.step()

    return D_x, D_G_z, errD


def adversarial_generative_train(netG, optimizerG, netD, prev, from_scales, m_noise, opt):
    netG.zero_grad()

    ##todo: I added!
    # train with fake
    curr = m_noise(from_scales[opt.curr_scale])
    fake = netG(curr, prev)
    ##end

    output = netD(fake.detach())
    errG = -1 * opt.lambda_adversarial * output.mean()
    errG.backward(retain_graph=True)

    # reconstruction loss (as appers in singan, doesn't work for my settings):
    # loss = nn.MSELoss()
    # prev_rec = concat_pyramid(Gs, source_scales, target_scales, source_scale, m_noise, m_image, opt)
    # rec_loss = loss(prev_rec.detach(), curr_scale)
    # rec_loss.backward(retain_graph=True)
    # rec_loss = rec_loss.detach()

    optimizerG.step()
    return errG

def cycle_consistency_loss(netGx, optimizerGx, x, x_scales, prev_x, netGy, optimizerGy, y, y_scales, prev_y, m_noise, opt):
    criterion = nn.L1Loss()
    optimizerGx.zero_grad()
    optimizerGy.zero_grad()

    #Gy(x):
    curr_x = m_noise(x_scales[opt.curr_scale])
    Gy_x = netGx(curr_x, prev_x)

    #Gx(Gy(x)):
    curr_yx = m_noise(Gy_x)
    Gx_Gy_x = netGy(curr_yx, prev_y)
    loss_x = opt.idx * criterion(Gx_Gy_x, x)
    loss_x.backward()

    # Gx(y):
    curr_y = m_noise(y_scales[opt.curr_scale])
    Gx_y = netGy(curr_y, prev_y)

    # Gy(Gx(y)):
    curr_xy = m_noise(Gx_y)
    Gy_Gx_y = netGx(curr_xy, prev_x)
    loss_y = opt.idy * criterion(Gy_Gx_y, y)
    loss_y.backward()

    loss = loss_x + loss_y
    optimizerGx.step()
    optimizerGy.step()

    return loss_x, loss_y, loss, (Gy_x, Gx_Gy_x, Gx_y, Gy_Gx_y)

def concat_pyramid(Gs, sources, m_noise, m_image, opt):
    if len(Gs) == 0:
        return torch.zeros_like(sources[0])

    G_z = sources[0]
    count = 0
    for G, source_curr, source_next in zip(Gs, sources, sources[1:]):
        G_z = G_z[:, :, 0:source_curr.shape[2], 0:source_curr.shape[3]]
        source_curr = m_noise(source_curr)
        G_z = m_image(G_z)
        G_z = G(source_curr, G_z.detach())
        # G_z = imresize(G_z, 1 / opt.scale_factor, opt)
        G_z = imresize_torch(G_z, 1 / opt.scale_factor, opt)
        G_z = G_z[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
        count += 1
    return G_z

def init_models(opt):
    # generator initialization:
    netG = models.Generator(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    return netD, netG

def norm_image(im):
    return (im + 1)/2

