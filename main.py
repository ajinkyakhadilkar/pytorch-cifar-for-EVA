'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from tqdm import tqdm

from .models import resnet
from . import utils


net = resnet.ResNet18()
grad_image_test = []
misclassified_images = []
misclassified_label = []
ground_truth = []

'''
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = utils.get_train_transforms()

transform_test = utils.get_test_transforms()

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def get_testloader(apply_transform=False):
  testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
      if apply_transform else transforms.ToTensor())
  return torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
  net = torch.nn.DataParallel(net)
  cudnn.benchmark = True

'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    global net
    pbar = tqdm(trainloader)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    sampled = False
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        if not sampled:
          grad_image_test = inputs
          sampled = True
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(desc= f'Epoch={epoch} Loss={train_loss/(batch_idx+1)} batch_id={batch_idx} Accuracy={100*correct/total:0.2f}%')


def test(epoch):
    global best_acc, net
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 1:
              grad_image_test = inputs
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            results = predicted.eq(targets)
            correct += results.sum().item()
            #Collecting missclassified images
            for i in range(0,len(targets)):
              if len(misclassified_images) < 10 and not results[i]:
                misclassified_images.append(inputs[i])
                ground_truth.append(targets[i])
                misclassified_label.append(predicted[i])

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss/(batch_idx+1), correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

    # Save checkpoint.
    '''
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    '''

def start_training():
  for epoch in range(start_epoch, start_epoch+20):
      train(epoch)
      test(epoch)
      scheduler.step()
