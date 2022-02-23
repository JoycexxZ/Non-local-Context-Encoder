import torch
import torch.nn as nn
import torch.nn.functional as F
import cudnn
from models.nlcen import *
from datasets.datasets import *
from models.loss import *
from utils import *


class Engine():
    def __init__(self, config):
        self.config = config

    def _adjust_learning_rate(self, optimizer):
        if lr > 1e-4:
            lr -= lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def train(self):
        model = Network_ResNet34()
        self.config = config_init(self.config)

        if torch.cuda.device_count() == 8:
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        elif torch.cuda.device_count() == 4:
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        else:
            model = model.cuda()

        cudnn.benchmark = True

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     betas=(self.config.first_momentum, self.config.second_momentum),
                                     weight_decay=self.config.weight_decay)

        dataloader = get_training_loader(self.config, self.config.batch_size, self.config.num_workers)

        for epoch in range(self.config.epochs):
            message(self.config, 'starting epoch {}...'.format(epoch))
            for i, (image, mask) in enumerate(dataloader):
                image = image.cuda()
                mask = mask.cuda()

                optimizer.zero_grad()

                out, p2_s, p3_s, p4_s, p5_s = model(image)
                loss = loss(p2_s, p3_s, p4_s, p5_s, out, mask, self.config.lamb)

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    message(self.config, '[%d/%d] Loss: %.4f' % (i, len(dataloader), loss.item()))

            self.adjust_learning_rate(optimizer)
            line(self.config)
            if epoch % 20 == 0 and self.config.out_to_folder == 'True':
                self.save_model(model, epoch)
            if epoch > self.config.epochs:
                break

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), '{}/model_{}.pth'.format(self.config.results_dir, epoch))
