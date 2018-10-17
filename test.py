import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import json

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from testdataset import VideoDataset
from p3d_model import P3D24

BatchSize = 1
nEpochs = 10  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 25 # Run on test set every nTestInterval epochs
snapshot = 4 # Store a model every snapshot epochs
lr = 1e-2 # Learning rate

dataset = 'future1' # Options: hmdb51 or ucf101
modelName = 'P3D24'
num_classes = 45
json_file = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
save_dir = '/home/shawntung/Desktop/FutureCamp/FutureCamp_ActionRecognitionData/run/run_0'

def test_model(dataset=dataset, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    model = P3D24(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 5 epochs
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=BatchSize,
                                 num_workers=4)
    test_size = len(test_dataloader.dataset)
    checkpoint = torch.load(
        os.path.join(save_dir, 'models', modelName + '_' + dataset + '_epoch-' + '23' + '.pth.tar'),
        map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_' + dataset + '_epoch-' + '23' + '.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])

    model.to(device)
    model.eval()
    start_time = timeit.default_timer()

    running_loss = 0.0
    running_corrects = 0

    for i, sample in enumerate(test_dataloader):
        inputs = Variable(sample[0]).to(device)
        labels = Variable(sample[1]).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)
        json_file.append({'label': preds[0].item()+1,'filename':sample[2][0]})
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print("[test]  Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")
    json_file2js = json.dumps(json_file, indent=2)
    fileObject = open('jsonfile.json','w')
    fileObject.write(json_file2js)
    fileObject.close()
if __name__ == "__main__":
    test_model()
