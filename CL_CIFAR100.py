import os
import numpy as np
import torch
import argparse

from main_DC import create_log_dir
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug, TensorDataset
import copy
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='LoDM', help='random/herding/DSA/DM/LoDSA/LoDM/')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=20, help='image(s) per class')
    parser.add_argument('--steps', type=int, default=5, help='5/10-step learning')
    parser.add_argument('--num_eval', type=int, default=3, help='evaluation number')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='/home/dataset/', help='dataset path')
    parser.add_argument('--cl_data_path', type=str, default='/home/CL_data/', help='dataset path')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True # augment images for all methods
    args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate' # for CIFAR10/100

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    args.save_path = args.cl_data_path + '/result/' + args.method + '/'
    logger = create_log_dir(args.save_path, 'log_steps_'+ str(args.steps) + '.txt')

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    ''' all training data '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    logger.info('==================================================================================')
    logger.info('method: ', args.method)
    results = np.zeros((args.steps, 5*args.num_eval))

    for seed_cl in range(5):
        num_classes_step = num_classes // args.steps
        np.random.seed(seed_cl)
        class_order = np.random.permutation(num_classes).tolist()
        logger.info('=========================================')
        logger.info('seed: '+str(seed_cl))
        logger.info('class_order: '+str(class_order))
        logger.info('augmentation strategy: '+str(args.dsa_strategy))
        logger.info('augmentation parameters: '+str(args.dsa_param.__dict__))

        if args.method == 'random':
            images_train_all = []
            labels_train_all = []
            for step in range(args.steps):
                classes_current = class_order[step * num_classes_step: (step + 1) * num_classes_step]
                images_train_all += [torch.cat([get_images(c, args.ipc) for c in classes_current], dim=0)]
                labels_train_all += [torch.tensor([c for c in classes_current for i in range(args.ipc)], dtype=torch.long, device=args.device)]

        elif args.method == 'herding':
            fname = os.path.join(args.cl_data_path,'cl_herding_CIFAR100_ConvNet_20ipc_%dsteps_seed%d.pt'%(args.steps, seed_cl))
            data = torch.load(fname, map_location='cpu')['data']
            images_train_all = [data[step][0] for step in range(args.steps)]
            labels_train_all = [data[step][1] for step in range(args.steps)]
            logger.info('use data: '+str(fname))

        elif args.method == 'DSA':
            fname = os.path.join(args.cl_data_path, 'cl_res_DSA_CIFAR100_ConvNet_20ipc_%dsteps_seed%d.pt'%(args.steps, seed_cl))
            data = torch.load(fname, map_location='cpu')['data']
            images_train_all = [data[step][0] for step in range(args.steps)]
            labels_train_all = [data[step][1] for step in range(args.steps)]
            logger.info('use data: ' +str(fname))

        elif args.method == 'DM':
            fname = os.path.join(args.cl_data_path, 'cl_DM_CIFAR100_ConvNet_20ipc_%dsteps_seed%d.pt'%(args.steps, seed_cl))
            data = torch.load(fname, map_location='cpu')['data']
            images_train_all = [data[step][0] for step in range(args.steps)]
            labels_train_all = [data[step][1] for step in range(args.steps)]
            logger.info('use data: ' +str(fname))

        elif args.method == 'LoDM':
            fname = os.path.join(args.cl_data_path, 'cl_LoDM_CIFAR100_ConvNet_80ipc_%dsteps_seed%d.pt' % (args.steps, seed_cl))
            data = torch.load(fname, map_location='cpu')['data']
            images_train_all = [data[step][0] for step in range(args.steps)]
            labels_train_all = [data[step][1] for step in range(args.steps)]
            logger.info('use data: ' + str(fname))

        else:
            logger.info('unknown method: %s'%args.method)

        for step in range(args.steps):
            logger.info('\n-----------------------------\nmethod %s seed %d step %d ' % (args.method, seed_cl, step))

            classes_seen = class_order[: (step+1)*num_classes_step]
            logger.info('classes_seen: '+str(classes_seen))

            ''' train data '''
            images_train = torch.cat(images_train_all[:step+1], dim=0).to(args.device)
            labels_train = torch.cat(labels_train_all[:step+1], dim=0).to(args.device)
            logger.info('train data size: '+str(images_train.shape))

            ''' test data '''
            images_test = []
            labels_test = []
            for i in range(len(dst_test)):
                lab = int(dst_test[i][1])
                if lab in classes_seen:
                    images_test.append(torch.unsqueeze(dst_test[i][0], dim=0))
                    labels_test.append(dst_test[i][1])

            images_test = torch.cat(images_test, dim=0).to(args.device)
            labels_test = torch.tensor(labels_test, dtype=torch.long, device=args.device)
            dst_test_current = TensorDataset(images_test, labels_test)
            testloader = torch.utils.data.DataLoader(dst_test_current, batch_size=256, shuffle=False, num_workers=0)

            logger.info('test set size: '+str(images_test.shape))

            ''' train model on the newest memory '''
            accs = []
            for ep_eval in range(args.num_eval):
                net_eval = get_network(args.model, channel, num_classes, im_size)
                net_eval = net_eval.to(args.device)
                img_syn_eval = copy.deepcopy(images_train.detach())
                lab_syn_eval = copy.deepcopy(labels_train.detach())

                _, acc_train, acc_test = evaluate_synset(ep_eval, net_eval, img_syn_eval, lab_syn_eval, testloader, args)
                del net_eval, img_syn_eval, lab_syn_eval
                gc.collect()  # to reduce memory cost
                accs.append(acc_test)
                results[step, seed_cl*args.num_eval + ep_eval] = acc_test
            logger.info('Evaluate %d random %s, mean = %.4f std = %.4f' % (len(accs), args.model, np.mean(accs), np.std(accs)))

    results_str = ''
    for step in range(args.steps):
        results_str += '& %.1f$\pm$%.1f  ' % (np.mean(results[step]) * 100, np.std(results[step]) * 100)

    logger.info('\n\n')
    logger.info('%d step learning %s perforamnce:'%(args.steps, args.method))
    logger.info(results_str)
    logger.info('Done')


if __name__ == '__main__':
    main()
