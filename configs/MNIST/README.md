## Hyper-parameters

## LoDM (SM)
- IPC=1

```
Hyper-parameters: 
'dataset': 'MNIST', 
'model': 'ConvNet',
'ipc': 7, 
'rank': 2, 
'eval_mode': 'SS', 
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
'outer_loop': 7, 
'inner_loop': 50, 
'device': 'cuda', 
'dsa': True
```


- IPC=10

```
Hyper-parameters: 
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 17, 
'rank': 8, 
'eval_mode': 'SS', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'none', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 17, 
'inner_loop': 50, 
'device': 'cuda', 
'dsa': False
```

- IPC=50

```
Hyper-parameters: 
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 88, 
'rank': 8, 
'eval_mode': 'SS', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'none', 
'dis_metric': 'ours', 
'method': 'DM', 
'outer_loop': 50, 
'inner_loop': 10, 
'device': 'cuda', 
'dsa': False
```

## LoDSA(SM)

- IPC=1

```
Hyper-parameters: 
'method': 'DSA', 
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 7, 
'rank': 2, 
'eval_mode': 'S', 
'num_exp': 2, 
'num_eval': 5, 
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
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 70, 
'rank': 2, 
'eval_mode': 'S', 
'num_exp': 2, 
'num_eval': 5, 
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

- IPC=50

```
Hyper-parameters: 
'method': 'DSA', 
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 88, 
'rank': 8, 
'eval_mode': 'S', 
'num_exp': 1, 
'num_eval': 1, 
'epoch_eval_train': 300, 
'Iteration': 1000, 
'lr_img': 0.1, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 
'dsa_strategy': 'None', 
'dis_metric': 'ours', 
'outer_loop': 50, 
'inner_loop': 10, 
'device': 'cuda', 
'dsa': True
```

## LoDC(SM)

- IPC=1

```
Hyper-parameters: 
'method': 'DC', 
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 8, 
'eval_mode': 'S', 
'num_exp': 2, 
'num_eval': 5, 
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
'dataset': 'MNIST', 
'model': 'ConvNet', 
'ipc': 70, 
'eval_mode': 'S', 
'num_exp': 2, 
'num_eval': 5, 
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

