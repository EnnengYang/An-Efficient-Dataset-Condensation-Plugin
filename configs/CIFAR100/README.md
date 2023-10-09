## Hyper-parameters

## LoDM (SM)

- IPC=1

```
Hyper-parameters:
'dataset': 'CIFAR100',
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
'dsa_strategy': 'color_crop_cutout_flip_scale_rotate',
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
'dataset': 'CIFAR100',
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


## LoDSA(SM)

- IPC=1

```
Hyper-parameters:
'method': 'DSA',
'dataset': 'CIFAR100',
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
'dsa_strategy': 'color_crop_cutout_flip_scale_rotate',
'dis_metric': 'ours',
'outer_loop': 1,
'inner_loop': 1,
'device': 'cuda',
'dsa': True
```

- IPC=10

```
Hyper-parameters:
'method': 'DSA',
'dataset': 'CIFAR100',
'model': 'ConvNet',
'ipc': 20,
'rank': 8,
'eval_mode': 'S',
'epoch_eval_train': 300,
'Iteration': 1000,
'lr_img': 0.1,
'lr_net': 0.01,
'batch_real': 256,
'batch_train': 256,
'init': 'noise',
'dsa_strategy': 'color_crop_cutout_flip_scale_rotate',
'dis_metric': 'ours',
'outer_loop': 10,
'inner_loop': 50,
'device': 'cuda',
'dsa': True
```


## LoDC(SM)

- IPC=1

```
Hyper-parameters:
'method': 'DC',
'dataset': 'CIFAR100',
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
'dataset': 'CIFAR100',
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
