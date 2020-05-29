import numpy as np
import torch.optim as optim
import time
import random
import torch
import h5py
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

torch.backends.cudnn.benchmark=True
device = torch.device("cuda")
use_cuda = torch.cuda.is_available()

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super(dataset_h5, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.n_depths, self.nx, self.ny, self.ch = self.file['img'].shape



    def __getitem__(self, index):
        images = self.file['img'][index,:, :, 0:3]
        labels = self.file['labels'][index, :]
        return images.astype('float32'), labels.astype('int32')

    def __len__(self):
        return self.n_depths

def DataLoader(h5_files_train, h5_files_test):
    train_loader = torch.utils.data.DataLoader(dataset_h5(h5_files_train),
                                                    batch_size=25, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset_h5(h5_files_test), batch_size=1)

    return train_loader, validation_loader

def next_batch(dataloader_iterator):

    next_batch_tensor = dataloader_iterator.next()
    return next_batch_tensor[0], next_batch_tensor[1]


torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
use_cuda = torch.cuda.is_available()
torch.cuda.manual_seed_all(random.randint(1, 10000))
torch.cuda.synchronize()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1_depth = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2).cuda()
        self.bn1_depth = nn.BatchNorm2d(64).cuda()
        self.relu_1 = nn.LeakyReLU(inplace=True)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_depth_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2).cuda()
        self.bn2_depth_2 = nn.BatchNorm2d(64).cuda()
        self.relu_2 = nn.LeakyReLU(inplace=True)
        self.fc_1 = nn.Linear(14400, 1024).cuda()
        self.fc_2 = nn.Linear(1024, 10).cuda()

    def forward(self, img_rgb):

        conv_1_out = self.conv1_depth(img_rgb)
        conv_1_out = self.bn1_depth(conv_1_out)
        conv_1_out = self.maxpool_1(self.relu_1(conv_1_out))

        conv_2_out = self.conv2_depth_2(conv_1_out)
        conv_2_out = self.bn2_depth_2(conv_2_out)
        conv_2_out = self.maxpool_2(self.relu_2(conv_2_out))

        flatten_input = torch.flatten(conv_2_out, 1)
        fc1 = self.fc_1(flatten_input)
        fc2 = self.fc_2(fc1)


        return torch.abs(fc2.cuda())


net = Net()
net.cuda()
print(net)
net.train()

h5_file = ['data_demo.h5']
criterion = nn.CrossEntropyLoss().cuda()

writer = SummaryWriter()

net = net.train()

f = h5py.File('/home/iman/myexperiments/pipe_segmentation/dataset/data_demo.h5', 'r')
images_h5 = np.asarray(list(f['img']))
labels_h5 = np.asarray(list(f['labels']))
s_ime = time.time()
optimizer = optim.Adam(list(net.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=0, amsgrad=False)
t= time.time()

for i_iter in range(0, 4):
    images_batch = torch.tensor(images_h5[i_iter * 25:(i_iter + 1) * 25].astype('float32'))
    labels_batch = torch.tensor(labels_h5[i_iter * 25:(i_iter + 1) * 25].astype('int32'))


    running_loss = 0.0



    images_reshaped = images_batch.permute(0, 3, 1, 2)
    outputs = net(images_reshaped.cuda())
    loss = criterion(outputs.cuda(), labels_batch.squeeze().long().cuda())
    running_loss += loss

e = time.time()
print('Time for training the network for 4 iterations with batch 25:', e-t)
writer.add_scalar('Loss/train', loss, epoch)

grid = torchvision.utils.make_grid(images_reshaped.cuda())
writer.add_image('images', grid, 0)
writer.add_graph(net.cuda(), images_reshaped.cuda())
writer.close()

