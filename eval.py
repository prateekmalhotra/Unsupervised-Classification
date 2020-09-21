"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from termcolor import colored
from utils.common_config import get_train_dataset, get_val_transformations, get_train_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank
from PIL import Image

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_exp', help='Location of config file')
FLAGS.add_argument('--model', help='Location where model is saved')
FLAGS.add_argument('--visualize_prototypes', default=True, action='store_true',
                    help='Show the prototpye for each cluster')
args = FLAGS.parse_args()

def merge_images(images, space=0, mean_img=None):
    num_images = images.shape[0]
    canvas_size = int(np.ceil(np.sqrt(num_images)))
    h = images.shape[1]
    w = images.shape[2]
    canvas = np.zeros((canvas_size * h + (canvas_size-1) * space,  canvas_size * w + (canvas_size-1) * space, 3), np.uint8)

    for idx in range(num_images):
        image = images[idx,:,:,:]
        if mean_img:
            image += mean_img
        i = idx % canvas_size
        j = idx // canvas_size
        min_val = np.min(image)
        max_val = np.max(image)
        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        canvas[j*(h+space):j*(h+space)+h, i*(w+space):i*(w+space)+w,:] = image
    return canvas

def main():

    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config)

    # Get dataset
    print(colored('Get train dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_train_dataset(config, transforms)
    dataloader = get_train_dataloader(config, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)
    print(model)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(args.model, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(state_dict)

    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])

    else:
        raise NotImplementedError

    # CUDA
    model.cuda()

    # Perform evaluation
    if config['setup'] in ['simclr', 'moco']:
        print(colored('Perform evaluation of the pretext task (setup={}).'.format(config['setup']), 'blue'))
        print('Create Memory Bank')
        if config['setup'] == 'simclr': # Mine neighbors after MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                    config['num_classes'], config['criterion_kwargs']['temperature'])

        else: # Mine neighbors before MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                    config['num_classes'], config['temperature'])
        memory_bank.cuda()

        print('Fill Memory Bank')
        fill_memory_bank(dataloader, model, memory_bank)

        print('Mine the nearest neighbors')
        for topk in [1, 5, 20]: # Similar to Fig 2 in paper
            _, acc = memory_bank.mine_nearest_neighbors(topk)
            print('Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(topk, 100*acc))


    elif config['setup'] in ['scan', 'selflabel']:
        print('Create Memory Bank')
        memory_bank = MemoryBank(len(dataset), 512, 13, 0.1)
        memory_bank.cuda()

        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        head = state_dict['head'] if config['setup'] == 'scan' else 0
        predictions, features = get_predictions(config, dataloader, model, return_features=True)

        print('Fill Memory Bank')
        memory_bank.update(features)
        print('Mine the nearest neighbors')

        topk = 20
        distances, indices = memory_bank.mine_nearest_neighbors(topk, distance_calc=True)
        some_indices = np.random.randint(0, high=len(dataset), size=100)

        for index in some_indices:
            visualize_neighbors(index, dataset, indices, distances)

        cluster_imgs = {}
        cluster_labels = np.array(predictions[0]['predictions'])

        for k, cluster_label in enumerate(cluster_labels):
            image = np.array(dataset.get_image(k))

            if cluster_label not in cluster_imgs:
                cluster_imgs[cluster_label] = [image]
            else:
                cluster_imgs[cluster_label].append(image)

        for cluster_label in cluster_imgs.keys():
            print("{} Images in cluster {}".format(len(cluster_imgs[cluster_label]), cluster_label))
            cluster_imgs[cluster_label] = np.array(cluster_imgs[cluster_label])
            plt.imsave("clusters/{}.jpg".format(cluster_label), merge_images(cluster_imgs[cluster_label]))

        # if args.visualize_prototypes:
        #     prototype_indices = get_prototypes(config, predictions[0], features, model)
        #     visualize_indices(prototype_indices, dataset)
    else:
        raise NotImplementedError

@torch.no_grad()
def get_prototypes(config, predictions, features, model, topk=10):
    import torch.nn.functional as F

    # Get topk most certain indices and pred labels
    print('Get topk')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]
    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim = 1)
    indices = torch.zeros((n_classes, topk))
    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k = topk, largest = True, sorted = True)
        indices[pred_id, :] = conf_idx

    # Get corresponding features
    selected_features = torch.index_select(features, dim=0, index=indices.view(-1).long())
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    # Get mean feature per class
    mean_features = torch.mean(selected_features, dim=1)

    # Get min distance wrt to mean
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    # Get final indices
    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).byte()
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices

def visualize_neighbors(index, dataset, indices, distances):
    imgs = []
    for k, idx in enumerate(indices[index]):
        if distances[index][k] < 30:
            imgs.append(np.array(dataset.get_image(idx)))

    if len(imgs) >= 2:
        plt.imsave("nearest_neighbors/{}.jpg".format(index), merge_images(np.array(imgs)))

def visualize_indices(indices, dataset):
    import matplotlib.pyplot as plt
    import numpy as np

    for idx in indices:
        img = dataset.get_image(idx)
        plt.figure()
        plt.axis('off')
        plt.imsave("{}.jpg".format(idx), img)

if __name__ == "__main__":
    main()
