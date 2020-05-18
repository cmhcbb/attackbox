import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import utils
import math
import random
import torch.nn.functional as F
import argparse
import os
import pdb
from scipy.fftpack import dct, idct
# parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
# parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
# parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
# parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
# parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
# parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
# parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
# parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
# parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
# parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
# parser.add_argument('--freq_dims', type=int, default=14, help='dimensionality of 2D frequency space')
# parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
# parser.add_argument('--stride', type=int, default=7, help='stride for block order')
# parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
# parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
# parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
# args = parser.parse_args()


def diagonal_order(image_size, channels):
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, :(image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(0, torch.LongTensor([i for i in range(i-1, -1, -1)]))
        order[i, (image_size - i):] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]

def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
        order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    return z.cuda()


def invert_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = [0,0,0]
        std = [1,1,1]
    elif dataset == 'mnist':
        mean = [0]
        std = [1]
    imgs_trans = imgs.clone()
    if len(imgs.size()) == 3:
        for i in range(imgs.size(0)):
            imgs_trans[i, :, :] = imgs_trans[i, :, :] * std[i] + mean[i]
    else:
        for i in range(imgs.size(1)):
            imgs_trans[:, i, :, :] = imgs_trans[:, i, :, :] * std[i] + mean[i]
    return imgs_trans


# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = [0,0,0]
        std = [1,1,1]
    elif dataset == 'mnist':
        mean = [0]
        std = [1]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor



def expand_vector(x, size, image_size):
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    z = torch.zeros(batch_size, 3, image_size, image_size)
    z[:, :, :size, :size] = x
    return z

def normalize(x):
    return apply_normalization(x, 'mnist')



def get_preds(model, x):
    output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    _, preds = output.data.max(1)
    return preds


class SimBA(object):
    def __init__(self,model):
        self.model = model

    def get_probs(self, x, y):
        output = self.model.predict_prob(normalize(torch.autograd.Variable(x.cuda())))
        probs = torch.index_select(torch.nn.Softmax()(output).data, 1, y)
        return torch.diag(probs)

    def dct_attack(self, images_batch, labels_batch, epsilon=0.2, freq_dims=14, stride=7, max_iters=10000, order='rand', targeted=False, pixel_attack=False, log_every=100):
        if order == 'rand':
            n_dims = 3 * freq_dims * freq_dims
        else:
            n_dims = 3 * image_size * image_size
        if max_iters > 0:
            max_iters = int(min(n_dims, max_iters))
        else:
            max_iters = int(n_dims)        


        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        # print(images_batch.size())
        # sample a random ordering for coordinates independently per batch element
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == 'diag':
            indices = diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch)
        # preds = get_preds(model, images_batch)
        preds = self.model.predict_label(images_batch, batch=True)
        # print(preds)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: block_idct(z, block_size=image_size)
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + trans(expand_vector(x[remaining_indices], expand_dims, image_size))).clamp(0, 1)
            perturbation = trans(expand_vector(x, expand_dims, image_size))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            # preds_next = get_preds(model, expanded)
            preds_next = self.model.predict_label(expanded, batch=True)

            preds[remaining_indices] = preds_next
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(expand_vector(x, expand_dims, image_size))).clamp(0, 1)
                # probs_k = get_probs(model, adv, labels_batch)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k-1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(expand_vector(left_vec, expand_dims, image_size))).clamp(0, 1)
            # left_probs = get_probs(model, adv, labels_batch[remaining_indices])
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                # print(left_probs, prev_probs[remaining_indices])
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(expand_vector(right_vec, expand_dims, image_size))).clamp(0, 1)
            # right_probs = get_probs(model, adv, labels_batch[remaining_indices])
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            prev_probs = prev_probs.cuda()
            # if (k + 1) % log_every == 0 or k == max_iters - 1:
            #     print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
            #             k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(expand_vector(x, expand_dims, image_size))).clamp(0, 1)
        # preds = get_preds(model, expanded)
        # preds = self.model.predict_label(expanded, batch=True)
        # if targeted:
        #     remaining = preds.ne(labels_batch)
        # else:
        #     remaining = preds.eq(labels_batch)
        # succs[:, max_iters-1] = ~remaining
        return expanded


    def __call__(self, input_xi, label_or_target, epsilon=0.2, eta=0.5, TARGETED=False):
        adv = self.dct_attack(input_xi, label_or_target, epsilon=epsilon, targeted=TARGETED)
        return adv  
