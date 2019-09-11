import argparse
import os
import logging
import sys
import itertools
import time

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.sku_dataset import SKUDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

import mAP
import detec_rate
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="sku", type=str,
                    help='Specify dataset type. Currently support sku and voc.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--test_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--net', default="mb3l-ssd-lite",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3s-ssd-lite or mb3l-ssd-lite.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument('--mb3_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV3')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--last_epoch', default=-1, type=int,
                    help='The index of last epoch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--checkbest_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
current_time = time.strftime('%y%m%d%H%M')
log_path = './Logs/'
log_name = log_path + current_time + '.log'
logfile = log_name
lfh = logging.FileHandler(logfile, mode='w')
lsh = logging.StreamHandler()
lfh.setLevel(logging.DEBUG)
lsh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s[%(lineno)d] - %(levelname)s : %(message)s')
lfh.setFormatter(formatter)
lsh.setFormatter(formatter)
logger.addHandler(lfh)
logger.addHandler(lsh)


args = parser.parse_args()
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # logger.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=50, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in enumerate(loader):

        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logger.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f} "
            )
            # logger.info(f'Average Train {debug_steps} Steps In {round((time.time()-train_time)/60, 2)} Mins')
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

# def res_test(loader, net, criterion, device):
def res_test(dataset, net, device):
    config = mobilenetv1_ssd_config
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    val_dataset = SKUDataset(dataset, transform=test_transform,
                             target_transform=target_transform, mode='1')
    loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)

    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for i, data in enumerate(loader):

        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i % 50 == 0:
            logger.info(f"Step: {i} in Test - loss : {loss}. ")

    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def test(loader, net, criterion, device, net_test, map_datas, test_file_path):

    net_state = net.state_dict()
    net_test.load_state_dict(net_state)
    net_test.to(device)
    class_names = map_datas.class_names[1:]

    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for i, data in enumerate(loader):

        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i % 50 == 0:
            logger.info(f"Step: {i} in Test - loss : {loss}. ")
            # logger.info(f"loss : {loss}. ")

    timer.start('map')
    aps = mAP.mAP(map_datas, net_test, device)
    print(f'MAP TIME :{timer.end("map")}')
    timer.start('detec reta')
    count, correct, wrong, miss = detec_rate.detec_rate(net_test, class_names, test_file_path, device)
    print(f'Detec Reta TIME :{timer.end("detec reta")}')


    return running_loss / num, running_regression_loss / num, running_classification_loss / num, aps, count, correct, wrong, miss
    # return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()
    writer = SummaryWriter(f'./Logs/{args.net}-{current_time}')
    logger.info(args)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) == 0:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = args.gpu_ids[0]
    if torch.cuda.device_count()==1 and DEVICE != 0:
        print(f"GPU device {DEVICE} is not supported.")
    if torch.cuda.device_count() < len(args.gpu_ids):
        print(f"GPU device num {len(args.gpu_ids)} is not supported.")
    logger.info(f"Use GPU {DEVICE}({args.gpu_ids}).")

    if args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3s-ssd-lite':
        create_net = lambda num: create_mobilenetv3_ssd_lite(num, model_mode="SMALL",
                                                             width_mult=args.mb3_width_mult, device=DEVICE)
        create_net_test = lambda num: create_mobilenetv3_ssd_lite(num, model_mode="SMALL",
                                                                  width_mult=args.mb3_width_mult,
                                                                  is_test=True, device=DEVICE)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3l-ssd-lite':
        create_net = lambda num: create_mobilenetv3_ssd_lite(num, model_mode="LARGE",
                                                             width_mult=args.mb3_width_mult, device=DEVICE)
        create_net_test = lambda num: create_mobilenetv3_ssd_lite(num, model_mode="LARGE",
                                                                  width_mult=args.mb3_width_mult,
                                                                  is_test=True, device=DEVICE)
        config = mobilenetv1_ssd_config
    else:
        logger.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Prepare Datasets
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logger.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            dataset = dataset
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'sku':
            dataset = SKUDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform, mode='0')
            label_file = os.path.join(args.checkpoint_folder, "sku-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        else:
            raise ValueError(f"Dataset tpye {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logger.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logger.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    logger.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == "sku":
        val_dataset = SKUDataset(args.datasets[0], transform=test_transform,
                                 target_transform=target_transform, mode='1')
        map_dataset = SKUDataset(args.datasets[0], mode='1')
    logger.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, args.batch_size,num_workers=args.num_workers,shuffle=False)

    logger.info("Build network.")
    net = create_net(num_classes)
    net_test = create_net_test(num_classes)
    best_val_loss = 5#0.8870
    best_map = 0#0.95
    last_epoch = args.last_epoch

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logger.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logger.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'initial_lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'initial_lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            ), 'initial_lr': base_net_lr}
        ]

    timer.start("Load Model")
    if args.resume:
        logger.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logger.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logger.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logger.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)
    net_test.to(DEVICE)
    net = torch.nn.DataParallel(net, args.gpu_ids)
    net_test = torch.nn.DataParallel(net, args.gpu_ids)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, )
    logger.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logger.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logger.info("Uses CosineAnnealingLR scheduler.")
        # t_max = args.t_max
        t_max = args.num_epochs
        scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)
    else:
        logger.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    model_save_folder = os.path.join(args.checkpoint_folder, f"{args.net}-{current_time}")
    if not os.path.exists(model_save_folder):
        os.mkdir(model_save_folder)
    logger.info(f"Model Save Folder In {model_save_folder}.")
    logger.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        print(f'Train:{epoch}')
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if epoch % args.checkbest_epochs == 0 or epoch == args.num_epochs - 1: #checkbest_epochsvalidation_epochs
            model_state_path = os.path.join(model_save_folder, f"{args.net}-state-Epoch-{epoch}.pth")
            if len(args.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), model_state_path)
                net.cuda(DEVICE)
            else:
                torch.save(net.cpu().state_dict(), model_state_path)
            logger.info(f"Saved model {model_state_path}")

            val_loss, val_regression_loss, val_classification_loss, aps, count, correct, wrong, miss = test(val_loader,
            # val_loss, val_regression_loss, val_classification_loss = test(val_loader,
                                                                                                            net,
                                                                                                            criterion,
                                                                                                            DEVICE,
                                                                                                            net_test,
                                                                                                            map_dataset,
                                                                                                            test_file_path=args.test_dataset)
            map = sum(aps) / len(aps)
            correct_rate = round(correct/count, 4)
            wrong_rate = round(wrong/count, 4)
            miss_rate = round(miss/count, 4)
            cla_name = val_dataset.class_names[1:]
            writer.add_scalars('Monitoring_Data/LOSS',
                               {'mAP': map, 'loss': val_loss, 'regression_loss': val_regression_loss,
                                'classification_loss': val_classification_loss, 'Correct detection rate': correct_rate,
                                'Detection rate': (1 - miss_rate)}, epoch)
            # writer.add_scalars('Monitoring_Data/AP', {class_name:ap for class_name, ap in zip(cla_name, aps)}, epoch)
            #
            logger.info(
                f"Epoch: {epoch}, " +
                # f"mAP: {map:.4f}({best_map}), " + # , {cla_name[0]}:{aps[0]}, {cla_name[1]}:{aps[1]}, {cla_name[2]}:{aps[2]}, {cla_name[3]}:{aps[3]}, {cla_name[4]}:{aps[4]}
                # f"Detection rate: {(1 - miss_rate)}, " +
                # f"Correct detection rate: {correct_rate}, " +
                f"Loss: {val_loss:.4f}({best_val_loss}), " + # Validation
                f"Regression Loss {val_regression_loss:.4f}, " +
                f"Classification Loss: {val_classification_loss:.4f}"
            )
            # if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            #     model_state_path = os.path.join(model_save_folder, f"{args.net}-state-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
            #     net.save(model_state_path)
            #     logger.info(f"Saved model {model_state_path}")
            # if val_loss < best_val_loss:
            #     best_loss_model_state_path = os.path.join(model_save_folder, f"{args.net}-best-loss-{round(val_loss, 4)}-Epoch-{epoch}.pth")
            #     net.save(best_loss_model_state_path)
            #     best_val_loss = val_loss
            #     logger.info(f'Saved best loss model {best_loss_model_state_path}')
            # if map > best_map:
            #     best_map_model_state_path = os.path.join(model_save_folder, f"{args.net}-best-mAP-{round(map, 4)}-Epoch-{epoch}.pth")
            #     net.save(best_map_model_state_path)
            #     best_map = map
            #     logger.info(f'Saved best map model {best_map_model_state_path}')
    writer.close()