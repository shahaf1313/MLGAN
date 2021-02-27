from config import get_arguments, post_config
import datetime
from data import CreateSrcDataLoader, CreateTrgDataLoader
from SinGAN.functions import denorm, colorize_mask
import numpy as np
import time
from constants import NUM_CLASSES, IGNORE_LABEL, trainId2label
from SinGAN.functions import compute_cm_batch_torch, compute_iou_torch, imresize_torch
import torch
import os
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from SemsegNetworks import CreateSemsegPyramidModel
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    multiscale_model = torch.load(opt.multiscale_model_path, map_location='cpu')
    opt.curr_scale = len(multiscale_model)
    opt.num_scales = len(multiscale_model)
    for scale in multiscale_model:
        scale.eval()
        scale.to(opt.device)

    source_train_loader = CreateSrcDataLoader(opt, 'train', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    opt.num_epochs = int(opt.num_steps/opt.epoch_size)
    target_val_loader = CreateTrgDataLoader(opt, 'val')

    feature_extractor, classifier, optimizer_fea, optimizer_cls = CreateSemsegPyramidModel(opt)
    scheduler_fea = torch.optim.lr_scheduler.StepLR(optimizer_fea, step_size=5,gamma=0.9)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=5, gamma=0.9)

    print('######### Network created #########')
    print('Architecture of Semantic Segmentation network:\n' + str(classifier) + str(feature_extractor))
    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, '%sGPU%d' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])))

    steps = 0
    print_int = 0
    save_pics_int = 0
    epoch_num = 1 if opt.semseg_model_epoch_to_resume > 0 else opt.semseg_model_epoch_to_resume + 1
    start = time.time()
    keep_training = True

    while keep_training:
        print('semeg train: starting epoch %d...' % (epoch_num))
        feature_extractor.train()
        classifier.train()

        for batch_num, (source_scales, source_label) in enumerate(source_train_loader):
            if steps > opt.num_steps:
                keep_training = False
                break

            # Move scale tensors to CUDA:
            for i in range(len(source_scales)):
                source_scales[i] = source_scales[i].to(opt.device)
            source_label = source_label.type(torch.long)
            source_label = source_label.to(opt.device)

            if opt.train_source:
                source_in_target = source_scales[-1]
            else:
                with torch.no_grad():
                    source_in_target = create_target_from_source(multiscale_model, source_scales, opt)

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()
            size = source_label.shape[-2:]
            pred_softs = classifier(feature_extractor(source_in_target), size)
            pred_labels = torch.argmax(pred_softs, dim=1)
            loss = criterion(pred_softs, source_label)
            loss.backward()
            optimizer_fea.step()
            optimizer_cls.step()
            opt.tb.add_scalar('TrainSemseg/loss', loss.item(), steps)


            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('train semseg:[%d/%d] ; elapsed time = %.2f secs per step' %
                      (print_int*opt.print_rate, opt.num_steps, elapsed/opt.print_rate))
                start = time.time()
                print_int += 1

            if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                s       = denorm(source_scales[-1][0])
                sit     = denorm(source_in_target[0])
                s_lbl   = colorize_mask(source_label[0])
                sit_lbl = colorize_mask(pred_labels[0])
                opt.tb.add_image('TrainSemseg/source', s, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_in_target', sit, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_label', s_lbl, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_in_target_label', sit_lbl, save_pics_int*opt.save_pics_rate)
                save_pics_int += 1

            steps += 1
        # Update LR:
        scheduler_fea.step()
        scheduler_cls.step()

        #Validation:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        iou, miou, cm = calculte_validation_accuracy(feature_extractor, classifier, target_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'Validtaion', iou, miou, epoch_num)
        print('train semseg: average accuracy of epoch #%d on target domain: mIoU = %2f' % (epoch_num, miou))

        # Save checkpoint:
        torch.save(feature_extractor, '%s/%s_%s_AdaptedToTarget_Epoch%d.pth' % (opt.out_,opt.model, 'featureExtractor', epoch_num))
        torch.save(classifier, '%s/%s_%s_AdaptedToTarget_Epoch%d.pth' % (opt.out_,opt.model, 'classifier', epoch_num))
        epoch_num += 1

    #Save final network:
    torch.save(feature_extractor, '%s/%s_%s_AdaptedToTarget_Epoch%d.pth' % (opt.out_,opt.model, 'featureExtractor', epoch_num))
    torch.save(classifier, '%s/%s_%s_AdaptedToTarget_Epoch%d.pth' % (opt.out_,opt.model, 'classifier', epoch_num))

    #Test:
    print('train semseg: starting final accuracy calculation...')
    iou, miou, cm = calculte_validation_accuracy(feature_extractor, classifier, target_val_loader, opt, epoch_num)
    save_epoch_accuracy(opt.tb, 'Test', iou, miou, epoch_num)
    opt.tb.close()
    print('Finished training.')

def create_target_from_source(Gs, sources, opt):
    G_n = torch.empty(1)
    for G, source_curr, source_next in zip(Gs, sources, sources[1:]):
        G_n = G(source_curr, G_n.detach())
        G_n = imresize_torch(G_n, 1 / opt.scale_factor)
        G_n = G_n[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
    # Last scale:
    G_n = Gs[-1](sources[-1], G_n.detach())
    return G_n

def save_epoch_accuracy(tb, set, iou, miou, epoch):
    if set == 'Validtaion':
        for i in range(NUM_CLASSES):
            tb.add_scalar('%sAccuracy/%s class accuracy' % (set, trainId2label[i].name), iou[i], epoch)
        tb.add_scalar('%sAccuracy/Accuracy History [mIoU]' % set, miou, epoch)
    print('================Epoch Acuuracy Summery================')
    for i in range(NUM_CLASSES):
        print('%s class accuracy: = %.2f' % (trainId2label[i].name, iou[i]))
    print('Average accuracy of test set on target domain: mIoU = %2f' % miou)
    print('======================================================')

def calculte_validation_accuracy(feature_extractor, classifier, target_val_loader, opt, epoch_num):
    feature_extractor.eval()
    classifier.eval()
    rand_samp_inds = np.random.randint(0, len(target_val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    for val_batch_num, (target_images, target_labels) in enumerate(target_val_loader):
        target_images = target_images.to(opt.device)
        target_labels = target_labels.to(opt.device)
        with torch.no_grad():
            size = target_labels.shape[-2:]
            pred_softs = classifier(feature_extractor(target_images), size)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, target_labels, IGNORE_LABEL, NUM_CLASSES)
            if val_batch_num in rand_batchs:
                t        = denorm(target_images[0])
                t_lbl    = colorize_mask(target_labels[0])
                pred_lbl = colorize_mask(pred_labels[0])
                opt.tb.add_image('ValidtaionEpoch%d/target' % epoch_num, t, val_batch_num)
                opt.tb.add_image('ValidtaionEpoch%d/target_label' % epoch_num, t_lbl, val_batch_num)
                opt.tb.add_image('ValidtaionEpoch%d/prediction_label' % epoch_num, pred_lbl, val_batch_num)
    iou, miou = compute_iou_torch(cm)
    return iou, miou, cm

if __name__ == "__main__":
    main()

