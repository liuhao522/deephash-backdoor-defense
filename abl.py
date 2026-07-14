import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import logging
import time
import yaml
from copy import deepcopy
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加必要的路径
sys.path.append('../')
sys.path.append(os.getcwd())

# 从原代码中导入必要的函数和类
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, BackdoorModelTrainer, all_acc, \
    general_plot_for_epoch, given_dataloader_test
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate


# ====================== ABL Defense Components ======================

class LGALoss(nn.Module):
    def __init__(self, gamma, criterion):
        super(LGALoss, self).__init__()
        self.gamma = gamma
        self.criterion = criterion
        return

    def forward(self, output, target):
        loss = self.criterion(output, target)
        loss_ascent = torch.sign(loss - self.gamma) * loss
        return loss_ascent


class FloodingLoss(nn.Module):
    def __init__(self, flooding, criterion):
        super(FloodingLoss, self).__init__()
        self.flooding = flooding
        self.criterion = criterion
        return

    def forward(self, output, target):
        loss = self.criterion(output, target)
        loss_ascent = (loss - self.flooding).abs() + self.flooding
        return loss_ascent


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.tuning_epochs:
        lr = args.lr
    else:
        lr = 0.01
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_loss_value(args, poisoned_data, model_ascent):
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      )

    for idx, (img, target, _, _, _) in tqdm(enumerate(example_data_loader, start=0)):
        img = img.to(args.device)
        target = target.to(args.device)

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))
    losses_record_arr = np.array(losses_record)
    logging.info(f'Top ten loss value: {losses_record_arr[losses_idx[:10]]}')

    return losses_idx


def isolate_data(args, result, losses_idx):
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = args.isolation_ratio
    perm = losses_idx[0: int(len(losses_idx) * ratio)]
    permnot = losses_idx[int(len(losses_idx) * ratio):]
    tf_compose = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
    train_dataset = result['bd_train'].wrapped_dataset
    data_set_without_tran = train_dataset
    data_set_isolate = result['bd_train']
    data_set_isolate.wrapped_dataset = data_set_without_tran
    data_set_isolate.wrap_img_transform = tf_compose

    data_set_other_without_tran = data_set_without_tran.copy()
    data_set_other = dataset_wrapper_with_transform(
        data_set_other_without_tran,
        tf_compose,
        None,
    )

    data_set_isolate.subset(perm)
    data_set_other.subset(permnot)

    logging.info('Finish collecting {} isolation examples: '.format(len(data_set_isolate)))
    logging.info('Finish collecting {} other examples: '.format(len(data_set_other)))

    return data_set_isolate, data_set_other


def learning_rate_finetuning(optimizer, epoch, args):
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def learning_rate_unlearning(optimizer, epoch, args):
    if epoch < args.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ====================== BadNet Attack ======================

def add_common_attack_args(parser):
    parser.add_argument('--attack', type=str, default='badnet')
    parser.add_argument('--attack_target', type=int, default=0,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str, default='all2one',
                        help='which type of label modification in backdoor attack')
    parser.add_argument('--pratio', type=float, default=0.1,
                        help='the poison rate')
    return parser


class BadNetAttack:
    def __init__(self, args):
        self.args = args

    def prepare_benign_data(self):
        """准备干净数据"""
        from utils.aggregate_block.dataset_and_transform_generate import get_transform
        from torchvision import datasets

        # 根据数据集选择适当的数据加载方式
        if self.args.dataset in ['mnist', 'cifar10', 'cifar100']:
            # 使用torchvision的标准数据集
            if self.args.dataset == 'mnist':
                train_dataset = datasets.MNIST(self.args.dataset_path, train=True, download=True)
                test_dataset = datasets.MNIST(self.args.dataset_path, train=False, download=True)
            elif self.args.dataset == 'cifar10':
                train_dataset = datasets.CIFAR10(self.args.dataset_path, train=True, download=True)
                test_dataset = datasets.CIFAR10(self.args.dataset_path, train=False, download=True)
            elif self.args.dataset == 'cifar100':
                train_dataset = datasets.CIFAR100(self.args.dataset_path, train=True, download=True)
                test_dataset = datasets.CIFAR100(self.args.dataset_path, train=False, download=True)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        # 获取数据变换
        train_transform = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                        train=True)
        test_transform = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                       train=False)

        # 创建包装的数据集
        clean_train_dataset = dataset_wrapper_with_transform(
            train_dataset, train_transform, None
        )
        clean_test_dataset = dataset_wrapper_with_transform(
            test_dataset, test_transform, None
        )

        return clean_train_dataset, clean_test_dataset, train_dataset, test_dataset

    def stage1_non_training_data_prepare(self):
        logging.info(f"Stage 1: Preparing poisoned data")

        # 准备干净数据
        clean_train_dataset_with_transform, clean_test_dataset_with_transform, train_dataset_without_transform, test_dataset_without_transform = self.prepare_benign_data()

        # 获取攻击的图像变换和标签变换
        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(self.args)
        bd_label_transform = bd_attack_label_trans_generate(self.args)

        # 生成毒化索引
        clean_train_targets = [label for _, label in train_dataset_without_transform]
        clean_test_targets = [label for _, label in test_dataset_without_transform]

        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=self.args.pratio,
        )

        # 生成毒化训练数据集
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{self.args.save_path}/bd_train_dataset",
        )

        train_transform = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                        train=True)
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_transform,
            None,
        )

        # 生成毒化测试数据集（用于计算ASR）
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{self.args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(np.where(test_poison_index == 1)[0])

        test_transform = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                       train=False)
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_transform,
            None,
        )

        self.stage1_results = (
            clean_train_dataset_with_transform,
            clean_test_dataset_with_transform,
            bd_train_dataset_with_transform,
            bd_test_dataset_with_transform,
            train_poison_index
        )

    def stage2_training(self):
        logging.info(f"Stage 2: Training backdoored model")

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform, \
        train_poison_index = self.stage1_results

        # 创建模型
        self.net = generate_cls_model(
            model_name=self.args.model,
            num_classes=self.args.num_classes,
            image_size=self.args.img_size[0],
        )

        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # 创建训练器
        trainer = BackdoorModelTrainer(self.net)

        criterion = argparser_criterion(self.args)
        optimizer, scheduler = argparser_opt_scheduler(self.net, self.args)

        from torch.utils.data.dataloader import DataLoader

        # 训练模型
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataset_with_transform, batch_size=self.args.batch_size, shuffle=True,
                       drop_last=True, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers),
            DataLoader(clean_test_dataset_with_transform, batch_size=self.args.batch_size, shuffle=False,
                       drop_last=False, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers),
            DataLoader(bd_test_dataset_with_transform, batch_size=self.args.batch_size, shuffle=False,
                       drop_last=False, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers),
            self.args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=self.args.frequency_save,
            save_folder_path=self.args.save_path,
            save_prefix='attack',
            amp=self.args.amp,
            prefetch=self.args.prefetch,
            non_blocking=self.args.non_blocking,
        )

        # 保存攻击结果
        attack_result = {
            'model': self.net.cpu().state_dict(),
            'bd_train': bd_train_dataset_with_transform,
            'bd_test': bd_test_dataset_with_transform,
            'clean_test': clean_test_dataset_with_transform,
        }

        torch.save(attack_result, f"{self.args.save_path}/attack_result.pt")
        logging.info(f"Attack completed. Results saved to {self.args.save_path}/attack_result.pt")

        return attack_result

    def run_attack(self):
        """运行完整的攻击流程"""
        self.stage1_non_training_data_prepare()
        return self.stage2_training()


# ====================== ABL Defense ======================

class ABLDefense:
    def __init__(self, args):
        self.args = args

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

    def pre_train(self, poisoned_data, model_ascent):
        """预训练阶段"""
        args = self.args
        agg = Metric_Aggregator()

        if "," in args.device:
            model_ascent = torch.nn.DataParallel(
                model_ascent,
                device_ids=[int(i) for i in args.device[5:].split(",")]
            )
            args.device = f'cuda:{model_ascent.device_ids[0]}'
            model_ascent.to(args.device)
        else:
            model_ascent.to(args.device)

        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        criterion = argparser_criterion(args).to(args.device)
        if args.gradient_ascent_type == 'LGA':
            criterion = LGALoss(args.gamma, criterion).to(args.device)
        elif args.gradient_ascent_type == 'Flooding':
            criterion = FloodingLoss(args.flooding, criterion).to(args.device)

        poisoned_data_loader = torch.utils.data.DataLoader(
            poisoned_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
        )

        logging.info('----------- Pre-training Start --------------')
        for epoch in range(args.tuning_epochs):
            logging.info(f"Pre-train Epoch {epoch + 1}/{args.tuning_epochs}")
            adjust_learning_rate(optimizer, epoch, args)

            model_ascent.train()
            for batch_idx, (img, target, _, _, _) in enumerate(poisoned_data_loader):
                img, target = img.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                output = model_ascent(img)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return model_ascent

    def train_unlearning(self, model_ascent, isolate_poisoned_data, isolate_other_data):
        """反学习阶段"""
        args = self.args
        agg = Metric_Aggregator()

        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=args.lr_unlearning_init,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        criterion = argparser_criterion(args).to(args.device)

        isolate_poisoned_data_loader = torch.utils.data.DataLoader(
            isolate_poisoned_data, batch_size=args.batch_size, shuffle=True
        )

        isolate_other_data_loader = torch.utils.data.DataLoader(
            isolate_other_data, batch_size=args.batch_size, shuffle=True
        )

        # 微调阶段（可选）
        if args.finetuning_ascent_model:
            logging.info('----------- Finetuning Start --------------')
            for epoch in range(args.finetuning_epochs):
                learning_rate_finetuning(optimizer, epoch, args)

                model_ascent.train()
                for batch_idx, (img, target, _, _, _) in enumerate(isolate_other_data_loader):
                    img, target = img.to(args.device), target.to(args.device)
                    optimizer.zero_grad()
                    output = model_ascent(img)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

        # 反学习阶段
        logging.info('----------- Unlearning Start --------------')
        for epoch in range(args.unlearning_epochs):
            learning_rate_unlearning(optimizer, epoch, args)

            model_ascent.train()
            for batch_idx, (img, target, _, _, _) in enumerate(isolate_poisoned_data_loader):
                img, target = img.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                output = model_ascent(img)
                loss = criterion(output, target)
                (-loss).backward()  # 梯度上升
                optimizer.step()

        return model_ascent

    def run_defense(self, attack_result):
        """运行完整的防御流程"""
        self.set_logger()
        args = self.args

        logging.info('----------- ABL Defense Start --------------')

        # 1. 预训练模型
        logging.info('Step 1: Pre-training model')
        model_ascent = generate_cls_model(args.model, args.num_classes)
        model_ascent = self.pre_train(attack_result['bd_train'], model_ascent)

        # 2. 计算损失并隔离数据
        logging.info('Step 2: Computing loss and isolating data')
        losses_idx = compute_loss_value(args, attack_result['bd_train'], model_ascent)
        isolate_poisoned_data, isolate_other_data = isolate_data(args, attack_result, losses_idx)

        # 3. 反学习
        logging.info('Step 3: Unlearning backdoor')
        defended_model = self.train_unlearning(model_ascent, isolate_poisoned_data, isolate_other_data)

        # 保存防御结果
        defense_result = {
            'model': defended_model.cpu().state_dict(),
        }

        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=defended_model.cpu().state_dict(),
            save_path=args.save_path,
        )

        logging.info(f"Defense completed. Results saved to {args.save_path}")
        return defense_result


# ====================== Main Pipeline ======================

def main():
    parser = argparse.ArgumentParser(description='BadNet Attack + ABL Defense Pipeline')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=42)

    # 攻击参数
    parser.add_argument('--attack_target', type=int, default=0)
    parser.add_argument('--pratio', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='./results')

    # 防御参数
    parser.add_argument('--tuning_epochs', type=int, default=5)
    parser.add_argument('--finetuning_epochs', type=int, default=2)
    parser.add_argument('--unlearning_epochs', type=int, default=3)
    parser.add_argument('--isolation_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_ascent_type', type=str, default='LGA', choices=['LGA', 'Flooding'])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--flooding', type=float, default=0.5)
    parser.add_argument('--finetuning_ascent_model', type=bool, default=True)
    parser.add_argument('--lr_unlearning_init', type=float, default=0.0001)

    args = parser.parse_args()

    # 设置随机种子
    fix_random(args.random_seed)

    # 获取数据集信息
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    args.log = os.path.join(args.save_path, 'logs')
    os.makedirs(args.log, exist_ok=True)

    # 步骤1: BadNet攻击
    print("=" * 50)
    print("Step 1: Running BadNet Attack")
    print("=" * 50)

    attack_args = deepcopy(args)
    attack_args.attack = 'badnet'
    attack_args.attack_label_trans = 'all2one'

    badnet_attack = BadNetAttack(attack_args)
    attack_result = badnet_attack.run_attack()

    # 步骤2: ABL防御
    print("=" * 50)
    print("Step 2: Running ABL Defense")
    print("=" * 50)

    defense_args = deepcopy(args)
    defense_args.save_path = os.path.join(args.save_path, 'defense')
    os.makedirs(defense_args.save_path, exist_ok=True)

    abl_defense = ABLDefense(defense_args)
    defense_result = abl_defense.run_defense(attack_result)

    print("=" * 50)
    print("Pipeline Completed Successfully!")
    print(f"Results saved to: {args.save_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()