# Hyper-parameters

## LoDM (SM)

- IPC=1

```
Hyper-parameters: 
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 8, 
'rank': 2, 
'eval_mode': 'SS', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 1, 
'inner_loop': 1, 
'device': 'cuda', 
'dsa': True
```

- IPC=10

```
Hyper-parameters: 
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 80, 
'rank': 2, 
'eval_mode': 'SS', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 10, 
'inner_loop': 50, 
'device': 'cuda', 
'dsa': True

```

- IPC=50

```
Hyper-parameters: 
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 200, 
'rank': 4, 
'eval_mode': 'SS', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1000.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 200, 
'inner_loop': 10, 
'device': 'cuda', 
'dsa': True
```




## LoDSA(SM)

- IPC=1

```
Hyper-parameters: 
'method': 'DSA', 
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 8, 
'rank': 2, 
'eval_mode': 'S', 
'epoch_eval_train': 300, 
'Iteration': 1000, 
'lr_img': 0.1, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 
'color_crop_cutout_flip_scale_rotate', 
'dis_metric': 'ours', 
'outer_loop': 1, 
'inner_loop': 1, 
'device': 'cuda', 
'dsa': True
```



## LoDC(SM)

- IPC=1

```
Hyper-parameters: 
'method': 'DC', 
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 8, 
'eval_mode': 'S', 
'epoch_eval_train': 300, 
'Iteration': 1000, 
'rank': 2, 
'lr_img': 0.1, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'None', 
'dis_metric': 'ours', 
'outer_loop': 1, 
'inner_loop': 1, 
'device': 'cuda', 
'dsa': False
```

- IPC=10

```
Hyper-parameters: 
'method': 'DC', 
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 80, 
'eval_mode': 'S', 
'epoch_eval_train': 300, 
'Iteration': 1000, 
'rank': 2, 
'lr_img': 0.1, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'None', 
'dis_metric': 'ours', 
'outer_loop': 10, 
'inner_loop': 50, 
'device': 'cuda', 
'dsa': False
```


## LoMTT(SM)

- IPC=1

```
'dataset': 'CIFAR10', 
'model': 'ConvNet', 
'ipc': 8, 
'rank': 2, 
'n_style': 5, 
'eval_mode': 'S', 
'eval_it': 100, 
'epoch_eval_train': 1000, 
'Iteration': 5000, 
'lr_img': 1000, 
'lr_lr': 1e-07, 
'lr_teacher': 0.01, 
'lr_style': 100, 
'lambda_club_content': 0.1, 
'lambda_contrast_content': 1, 
'lambda_cls_content': 1, 
'lambda_likeli_content': 1, 
'lr_init': 0.01, 
'batch_real': 256, 
'batch_syn': 80, 
'batch_train': 256, 
'pix_init': 'noise', 
'dsa': True, 
'dsa_strategy': 
'color_crop_cutout_flip_scale_rotate', 
'expert_epochs': 2, 
'syn_steps': 50, 
'max_start_epoch': 5, 
'zca': False, 
'load_all': False, 
'no_aug': False, 
'max_files': None, 
'max_experts': None, 
'force_save': False, 
'device': 'cuda', 
'dc_aug_param': None, 
'zca_trans': None, '
distributed': False
```


# Cross-architecture testing performance


- AlexNet (IPC=10, SM)

```
Hyper-parameters: 
'dataset': 'CIFAR10', 
'model': 'AlexNetBN', 
'ipc': 80, 
'rank': 2, 
'eval_mode': 'M', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 10, 
'inner_loop': 50, 
'device': 'cuda', 
'dsa': True
```

- ConvNet (IPC=10, SM)

```
Hyper-parameters: 
'dataset': 'CIFAR10', 
'model': 'ConvNetBN', 
'ipc': 80, 
'rank': 2, 
'eval_mode': 'M', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 
'color_crop_cutout_flip_scale_rotate', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 10, 
'inner_loop': 50, 
'device': 'cuda', 
'dsa': True
```