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
    Gs = []
    Ds = []

    while scale_num < opt.stop_scale + 1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        D_curr, G_curr = init_models(opt)
        if (nfc_prev == opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
        D_curr, G_curr = nn.DataParallel(D_curr, device_ids=[0,1,2,3,4,5,6]), nn.DataParallel(G_curr, device_ids=[0,1,2,3,4,5,6])

        D_curr, G_curr = train_single_scale(D_curr, G_curr, Gs, opt)

        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Ds.append(D_curr)
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(Ds, '%s/Ds.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def train_single_scale(netD, netG, Gs, opt):
    tb = SummaryWriter('./runs/scale%d/%s' % (len(Gs), datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S')))
    curr_scale = len(Gs)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    discriminator_steps = 0
    generator_steps = 0
    for batch_num, ((src_img, src_lbl, src_shapes, src_names), (trg_img, trg_lbl, trg_shapes, trg_names)) in enumerate(zip(opt.source_loaders[curr_scale], opt.target_loaders[curr_scale])):
        if batch_num > opt.niter:
            break

        source_scales = functions.creat_reals_pyramid(src_img, curr_scale, opt)
        target_scales = functions.creat_reals_pyramid(trg_img, curr_scale, opt)
        source_scale = source_scales[len(Gs)]
        target_scale = target_scales[len(Gs)]
        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        real = target_scale
        for j in range(opt.Dsteps):
            # train with real image
            netD.zero_grad()
            output = netD(real).to(opt.device)
            # D_real_map = output.detach()
            errD_real = -output.mean()  # -a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fakזe
            prev = concat_pyramid(Gs, source_scales, target_scales, m_noise, m_image, opt)
            prev = m_image(prev)
            curr = m_noise(source_scale)
            fake = netG(curr, prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

            tb.add_scalar('Scale%d/DiscriminatorLoss' % len(Gs), errD.item(), discriminator_steps)
            tb.add_scalar('Scale%d/DiscriminatorRealImagesLoss' % len(Gs), D_x, discriminator_steps)
            tb.add_scalar('Scale%d/DiscriminatorFakeImagesLoss' % len(Gs), D_G_z, discriminator_steps)

            discriminator_steps += 1

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake.detach())
            errG = -output.mean()
            errG.backward(retain_graph=True)

            #reconstruction loss (as appers in singan, doesn't work for my settings):
            # loss = nn.MSELoss()
            # prev_rec = concat_pyramid(Gs, source_scales, target_scales, source_scale, m_noise, m_image, opt)
            # rec_loss = loss(prev_rec.detach(), curr_scale)
            # rec_loss.backward(retain_graph=True)
            # rec_loss = rec_loss.detach()

            optimizerG.step()

            tb.add_scalar('Scale%d/GeneratorLoss' % len(Gs), errG.item(), generator_steps)
            # tb.add_scalar('Reconstruction loss at Scale %d' % len(Gs), rec_loss.item(), generator_steps)
            generator_steps += 1


        if (batch_num+1) % 25 == 0  or batch_num == 0:
            print('scale %d:[%d/%d]' % (len(Gs), batch_num+1, len(opt.source_loaders[curr_scale])))

        if (batch_num+1) % 250 == 0 or batch_num == 0:
            tb.add_image('Scale%d/fake_sit' % (len(Gs)), (fake[0]+1)/2, batch_num+1)
            # tb.add_images('Scale%d/reconstruction' % (len(Gs)), (prev_rec+1)/2, batch_num+1)
            tb.add_image('Scale%d/source' % (len(Gs)), (source_scale[0]+1)/2, batch_num+1)
            tb.add_image('Scale%d/target' % (len(Gs)), (target_scale[0]+1)/2, batch_num+1)
            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netD.module, netG.module, opt)
    tb.close()
    return netD.module, netG.module


def concat_pyramid(Gs, sources, targets, m_noise, m_image, opt):
    if len(Gs) == 0:
        return torch.zeros_like(sources[0])

    G_z = sources[0]
    count = 0
    for G, source_curr, source_next, target_curr, target_next in zip(Gs, sources, sources[1:], targets, targets[1:]):
        G_z = G_z[:, :, 0:source_curr.shape[2], 0:source_curr.shape[3]]
        source_curr = m_noise(source_curr)
        G_z = m_image(G_z)
        # z_in = 1 * source_curr + G_z #todo: concat source scale and prev scale insted of adding them! (do it inside G rather than here in this line)!!
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
