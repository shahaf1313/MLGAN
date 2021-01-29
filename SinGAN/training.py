import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
from SinGAN.imresize import imresize
import datetime
from torch.utils.tensorboard import SummaryWriter

def train(opt):
    scale_num = 0
    nfc_prev = 0
    Gst, Gts = [], []
    Dst, Dts = [], []

    while scale_num < opt.stop_scale + 1:
        opt.curr_scale = scale_num
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        Dst_curr, Gst_curr = init_models(opt)
        Dts_curr, Gts_curr = init_models(opt)
        if (nfc_prev == opt.nfc and opt.curr_scale > 1):
            Dst_curr.load_state_dict(torch.load('%s/%d/netDst.pth' % (opt.out_, scale_num - 1)))
            Gst_curr.load_state_dict(torch.load('%s/%d/netGst.pth' % (opt.out_, scale_num - 1)))
            Dts_curr.load_state_dict(torch.load('%s/%d/netDts.pth' % (opt.out_, scale_num - 1)))
            Gts_curr.load_state_dict(torch.load('%s/%d/netGts.pth' % (opt.out_, scale_num - 1)))
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
    return

def train_single_scale(netDst, netGst, netDts, netGts, Gst: list, Gts: list, Dst: list, Dts: list, opt):
        opt.tb = SummaryWriter('./runs/scale%d/%s' % (opt.curr_scale, datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S')))
        # setup optimizers and schedulers:
        optimizerDst = optim.Adam(netDst.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGst = optim.Adam(netGst.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerDts = optim.Adam(netDts.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGts = optim.Adam(netGts.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        schedulerDst = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerDst, milestones=[1600], gamma=opt.gamma)
        schedulerGst = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerGst, milestones=[1600], gamma=opt.gamma)
        schedulerDts = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerDts, milestones=[1600], gamma=opt.gamma)
        schedulerGts = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerGts, milestones=[1600], gamma=opt.gamma)

        discriminator_steps = 0
        generator_steps = 0
        for batch_num, ((src_img, src_lbl, src_shapes, src_names), (trg_img, trg_lbl, trg_shapes, trg_names)) in enumerate(zip(opt.source_loaders[opt.curr_scale], opt.target_loaders[opt.curr_scale])):
            # if batch_num > 3:
            #     break
            source_scales = functions.creat_reals_pyramid(src_img, opt)
            target_scales = functions.creat_reals_pyramid(trg_img, opt)

            pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
            pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
            m_noise = nn.ZeroPad2d(int(pad_noise))
            m_image = nn.ZeroPad2d(int(pad_image))

            ############################
            # (1) Update D networks: maximize D(x) + D(G(z))
            ###########################
            fake_sit, fake_tis = None, None
            for j in range(opt.Dsteps):
                #train discriminator networks between domains (S->T, T->S)

                #S -> T:
                real_images = target_scales[opt.curr_scale]
                D_x, D_G_z, errD = adversarial_disciriminative_train(netDst, optimizerDst, netGst, Gst, real_images, source_scales, m_noise, m_image, opt)
                opt.tb.add_scalar('Scale%d/ST/DiscriminatorLoss' % opt.curr_scale, errD.item(), discriminator_steps)
                opt.tb.add_scalar('Scale%d/ST/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x, discriminator_steps)
                opt.tb.add_scalar('Scale%d/ST/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z, discriminator_steps)


                # T -> S:
                real_images = source_scales[opt.curr_scale]
                D_x, D_G_z, errD = adversarial_disciriminative_train(netDts, optimizerDts, netGts, Gts, real_images, target_scales, m_noise, m_image, opt)
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
                errG, fake_sit = adversarial_generative_train(netGst, optimizerGst, netDst, Gst, source_scales, m_noise, m_image, opt)
                opt.tb.add_scalar('Scale%d/ST/GeneratorAdversarialLoss' % opt.curr_scale, errG.item(), generator_steps)

                # T -> S:
                errG, fake_tis = adversarial_generative_train(netGts, optimizerGts, netDts, Gts, target_scales, m_noise, m_image, opt)
                opt.tb.add_scalar('Scale%d/TS/GeneratorAdversarialLoss' % opt.curr_scale, errG.item(), generator_steps)

                # Identity Loss:
                loss_x, loss_y, loss_id = cycle_consistency_loss(netGst, optimizerGst, source_scales[opt.curr_scale], source_scales, Gst,
                                                              netGts, optimizerGts, target_scales[opt.curr_scale], source_scales, Gts, m_image, m_noise, opt)
                opt.tb.add_scalar('Scale%d/Identity/LossX' % opt.curr_scale, loss_x.item(), generator_steps)
                opt.tb.add_scalar('Scale%d/Identity/LossY' % opt.curr_scale, loss_y.item(), generator_steps)
                opt.tb.add_scalar('Scale%d/Identity/Loss' % opt.curr_scale, loss_id.item(), generator_steps)

                generator_steps += 1

            if (batch_num + 1) % 25 == 0 or batch_num == 0:
                print('scale %d:[%d/%d]' % (opt.curr_scale, batch_num + 1, len(opt.source_loaders[opt.curr_scale])))

            if (batch_num + 1) % 250 == 0 or batch_num == 0:
                opt.tb.add_image('Scale%d/fake_sit' % opt.curr_scale, (fake_sit[0] + 1) / 2, batch_num + 1)
                opt.tb.add_image('Scale%d/fake_tis' % opt.curr_scale, (fake_tis[0] + 1) / 2, batch_num + 1)
                opt.tb.add_image('Scale%d/source' % opt.curr_scale, (source_scales[opt.curr_scale][0] + 1) / 2, batch_num + 1)
                opt.tb.add_image('Scale%d/target' % opt.curr_scale, (target_scales[opt.curr_scale][0] + 1) / 2, batch_num + 1)
                # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
                # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
                # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
                # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
                # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
                # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

            schedulerDst.step()
            schedulerGst.step()
            schedulerDts.step()
            schedulerGts.step()

        opt.tb.close()
        if (len(opt.gpus) > 1):
            functions.save_networks(netDst.module, netGst.module, netDts.module, netGts.module, opt)
            return netDst.module, netGst.module, netDts.module, netGts.module
        else:
            functions.save_networks(netDst, netGst, netDts, netGts, opt)
            return netDst, netGst, netDts, netGts

def adversarial_disciriminative_train(netD, optimizerD, netG, Gs, real_images, from_scales, m_noise, m_image, opt):
    # train with real image
    netD.zero_grad()
    output = netD(real_images).to(opt.device)
    errD_real = -output.mean()  # -a
    errD_real.backward(retain_graph=True)
    D_x = -errD_real.item()

    # train with fake
    prev = concat_pyramid(Gs, from_scales, m_noise, m_image, opt)
    prev = m_image(prev)
    curr = m_noise(from_scales[opt.curr_scale])
    fake_images = netG(curr, prev)
    output = netD(fake_images.detach())
    errD_fake = output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = output.mean().item()

    gradient_penalty = functions.calc_gradient_penalty(netD, real_images, fake_images, opt.lambda_grad, opt.device)
    gradient_penalty.backward()

    errD = errD_real + errD_fake + gradient_penalty
    optimizerD.step()

    return D_x, D_G_z, errD


def adversarial_generative_train(netG, optimizerG, netD, Gs, from_scales, m_noise, m_image, opt):
    netG.zero_grad()

    ##I added!
    # train with fake
    prev = concat_pyramid(Gs, from_scales, m_noise, m_image, opt)
    prev = m_image(prev)
    curr = m_noise(from_scales[opt.curr_scale])
    fake = netG(curr, prev)
    ##end

    output = netD(fake.detach())
    errG = -output.mean()
    errG.backward(retain_graph=True)

    # reconstruction loss (as appers in singan, doesn't work for my settings):
    # loss = nn.MSELoss()
    # prev_rec = concat_pyramid(Gs, source_scales, target_scales, source_scale, m_noise, m_image, opt)
    # rec_loss = loss(prev_rec.detach(), curr_scale)
    # rec_loss.backward(retain_graph=True)
    # rec_loss = rec_loss.detach()

    optimizerG.step()
    return errG, fake

def cycle_consistency_loss(netGx, optimizerGx, x, x_scales, Xs, netGy, optimizerGy, y, y_scales, Ys, m_image, m_noise, opt):
    criterion = nn.L1Loss()
    optimizerGx.zero_grad()
    optimizerGy.zero_grad()

    #Gy(x):
    prev_x = concat_pyramid(Xs, x_scales, m_noise, m_image, opt)
    prev_x = m_image(prev_x)
    curr_x = m_noise(x_scales[opt.curr_scale])
    Gy_x = netGy(curr_x, prev_x)

    #Gx(Gy(x)):
    prev_yx = concat_pyramid(Ys, y_scales, m_noise, m_image, opt)
    prev_yx = m_image(prev_yx)
    curr_yx = m_noise(Gy_x)
    Gx_Gy_x = netGx(curr_yx, prev_yx)
    loss_x = opt.idx * criterion(Gx_Gy_x, x)
    loss_x.backward()

    # Gx(y):
    prev_y = concat_pyramid(Ys, y_scales, m_noise, m_image, opt)
    prev_y = m_image(prev_y)
    curr_y = m_noise(y_scales[opt.curr_scale])
    Gx_y = netGx(curr_y, prev_y)

    # Gy(Gx(y)):
    prev_xy = concat_pyramid(Xs, x_scales, m_noise, m_image, opt)
    prev_xy = m_image(prev_xy)
    curr_xy = m_noise(Gx_y)
    Gy_Gx_y = netGx(curr_xy, prev_xy)
    loss_y = opt.idy * criterion(Gy_Gx_y, y)
    loss_y.backward()

    loss = loss_x + loss_y
    optimizerGx.step()
    optimizerGy.step()

    return loss_x, loss_y, loss

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
        G_z = imresize(G_z, 1 / opt.scale_factor, opt)
        G_z = G_z[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
        count += 1
    return G_z

def init_models(opt):
    # generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
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
