## Hyper-parameters

## LoDM (SM)
- IPC=1
```
'dataset': 'TinyImageNet', 
'model': 'ConvNet', 
'ipc': 8, 
'rank': 4, 
'eval_mode': 'SS', 
'epoch_eval_train': 1000, 
'Iteration': 20000, 
'lr_img': 1.0, 
'lr_net': 0.01, 
'batch_real': 256, 
'batch_train': 256, 
'init': 'noise', 'dsa_strategy': 'color_crop_cutout_flip_scale_rotate', 
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
'dataset': 'TinyImageNet', 
'model': 'ConvNet', 
'ipc': 80, 
'rank': 4, 
'eval_mode': 'SS', 
'epoch_eval_train': 5000, 
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



