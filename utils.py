'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import albumentations as A

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

from torch.functional import Tensor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as T
from numpy import transpose
import matplotlib.pyplot as plt
import numpy as np

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
term_width = 120

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_cifar10_trainloader_with_transform(transforms):
    trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)


def get_cifar10_test_loader_with_transform(transforms):
    testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


def get_train_transforms(transforms_list=None):
  if not transforms_list:
    return transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
  else:
    return A.Compose(transforms_list)


def get_test_transforms():
  return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])

def show_gradcam(inp_image, model, plot=True):
  target_layers = [model.module.layer4[-1]]

  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

  targets = [ClassifierOutputTarget(9)]

  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=inp_image, targets=targets)

  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  rgb_img = (
      inp_image.cpu().squeeze().permute(1,2,0).detach().numpy()
  )
  print(grayscale_cam.shape)
  print(rgb_img.shape)
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  if plot:
    plt.imshow(visualization)
  return visualization

def get_misclassified_images(misclassified_images, misclassified_labels, ground_truth, classes):
  output_line=''
  print('Misclassification \n')
  fig, axs = plt.subplots(2, 5, figsize=(15, 8))
  for i in range(0, 10):
    axs[int(i/5)][int(i%5)].imshow(torch.stack(misclassified_images).cpu().detach().numpy()[i].squeeze())
    axs[int(i/5)][int(i%5)].set_title("Incorrect label: " + classes[misclassified_labels[i].item()] + "\n Correct label: " + classes[ground_truth[i].item()])
    axs[int(i/5)][int(i%5)].axis('off')
    #plt.subplot(2, 5, i+1)
    #plt.axis('off')
    #axs[i%5][int(i/5)].set_title("Incorrect label: " + misclassified_labels[i] + " Correct label: " + ground_truth[i])
    #plt.suptitle("Incorrect label: " + misclassified_labels[i].item() + " Correct label: " + ground_truth[i.item()])
    #plt.imshow(torch.stack(misclassified_images).cpu().detach().numpy()[i].squeeze())
    #plt.set_title('Incorrect Label: '+ misclassified_labels[i].item() + 'Correct Label: ' + ground_truth[i].item())

  print('Correct Labels: ')
  for i in range(0, len(misclassified_images)):
    output_line += str(ground_truth[i].item()) + ' '
    if i ==int((len(misclassified_images)-1)/2) or i==len(misclassified_images)-1:
      print(output_line + '\n')
      output_line = ''

  print('Prediction : ')
  for i in range(0, len(misclassified_images)):
    output_line += str(misclassified_labels[i].item()) + ' '
    if i ==int((len(misclassified_images)-1)/2) or i==len(misclassified_images)-1:
      print(output_line + '\n')
      output_line = ''

  #cleanup lists
  missclassified = []
  expected_label = []
  missclassified_value = []

  print('Images')


def get_gradcam_of_misclassified_images(misclassified_images, misclassified_labels, ground_truth, model, classes):
  output_line=''
  print('Misclassification \n')
  grads_misclassified_images = [torch.Tensor(show_gradcam(misclassified_image, model, plot=False)) for misclassified_image in misclassified_images]
  fig, axs = plt.subplots(2, 5, figsize=(15, 8))
  for i in range(0, 10):
    axs[int(i/5)][int(i%5)].imshow(torch.stack(grads_misclassified_images).cpu().detach().numpy()[i].squeeze())
    axs[int(i/5)][int(i%5)].set_title("Incorrect label: " + classes[misclassified_labels[i].item()] + "\n Correct label: " + classes[ground_truth[i].item()])
    axs[int(i/5)][int(i%5)].axis('off')
    #plt.subplot(2, 5, i+1)
    #plt.axis('off')
    #axs[i%5][int(i/5)].set_title("Incorrect label: " + misclassified_labels[i] + " Correct label: " + ground_truth[i])
    #plt.suptitle("Incorrect label: " + misclassified_labels[i].item() + " Correct label: " + ground_truth[i.item()])
    #plt.imshow(torch.stack(misclassified_images).cpu().detach().numpy()[i].squeeze())
    #plt.set_title('Incorrect Label: '+ misclassified_labels[i].item() + 'Correct Label: ' + ground_truth[i].item())

  print('Correct Labels: ')
  for i in range(0, len(grads_misclassified_images)):
    output_line += str(ground_truth[i].item()) + ' '
    if i ==int((len(grads_misclassified_images)-1)/2) or i==len(grads_misclassified_images)-1:
      print(output_line + '\n')
      output_line = ''

  print('Prediction : ')
  for i in range(0, len(grads_misclassified_images)):
    output_line += str(misclassified_labels[i].item()) + ' '
    if i ==int((len(grads_misclassified_images)-1)/2) or i==len(grads_misclassified_images)-1:
      print(output_line + '\n')
      output_line = ''

  #cleanup lists
  missclassified = []
  expected_label = []
  missclassified_value = []

  print('Images')


def plot_train_and_test_losses(train_losses, test_losses):
    fig, axs = plt.subplots(1,2,figsize=(15,10))
    axs[0].plot(train_losses)
    axs[0].set_xlabel('Loss')
    axs[0].set_ylabel('Iterations')
    axs[0].set_title("Training Loss")
    axs[1].plot(test_losses)
    axs[1].set_xlabel('Loss')
    axs[1].set_ylabel('Iterations')
    axs[1].set_title("Validation Loss")

