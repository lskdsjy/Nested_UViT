import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit': 
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    
    elif name == 'u2vit_t2i_old':
        from libs.u2vit_t2i_old import UViT # updated U-ViT
        return UViT(**kwargs)

    elif name == 'u2vit_t2i_v1':
        from libs.u2vit_t2i import NestedUViT_1  # with 1 block each

        # Initialize the NestedUViT model with the checkpoint path and additional arguments
        checkpoint_path = "/scratch/laks/U-ViT/workdir/medimgs/depth=16/ckpts/290000.ckpt/nnet_ema.pth"  # Update with the actual path
        model = NestedUViT_1(checkpoint_path, **kwargs)  # Initialize NestedUViT model with loaded checkpoint

        print("UViT weights loaded successfully into NestedUViT.")
        return model

    elif name == 'u2vit_t2i_v2':
        from libs.u2vit_t2i import NestedUViT_2  # with 2 blocks each

        # Initialize the NestedUViT model with the checkpoint path and additional arguments
        checkpoint_path = "/scratch/laks/U-ViT/workdir/medimgs/depth=16/ckpts/290000.ckpt/nnet_ema.pth"  # Update with the actual path
        model = NestedUViT_2(checkpoint_path, **kwargs)  # Initialize NestedUViT model with loaded checkpoint

        print("UViT weights loaded successfully into NestedUViT.")
        return model

    elif name == 'u2vit_t2i_v2S':
        from libs.u2vit_t2i import NestedUViT_2S  # with 2 blocks each - concat side outputs

        # Initialize the NestedUViT model with the checkpoint path and additional arguments
        checkpoint_path = "/scratch/laks/U-ViT/workdir/medimgs/depth=16/ckpts/290000.ckpt/nnet_ema.pth"  # Update with the actual path
        model = NestedUViT_2S(checkpoint_path, **kwargs)  # Initialize NestedUViT model with loaded checkpoint

        print("UViT weights loaded successfully into NestedUViT.")
        return model

    elif name == 'u2vit_t2i_v3':
        from libs.u2vit_t2i import NestedUViT_3  # with 3 block each

        # Initialize the NestedUViT model with the checkpoint path and additional arguments
        checkpoint_path = "/scratch/laks/U-ViT/workdir/medimgs/depth=16/ckpts/290000.ckpt/nnet_ema.pth"  # Update with the actual path
        model = NestedUViT_3(checkpoint_path, **kwargs)  # Initialize NestedUViT model with loaded checkpoint

        print("UViT weights loaded successfully into NestedUViT.")
        return model

    elif name == 'u2vit_t2i_v4':
        from libs.u2vit_t2i import NestedUViT_4  # with 4 blocks each

        # Initialize the NestedUViT model with the checkpoint path and additional arguments
        checkpoint_path = "/scratch/laks/U-ViT/workdir/medimgs/depth=16/ckpts/290000.ckpt/nnet_ema.pth"  # Update with the actual path
        model = NestedUViT_4(checkpoint_path, **kwargs)  # Initialize NestedUViT model with loaded checkpoint

        print("UViT weights loaded successfully into NestedUViT.")
        return model

    elif name == 'u2vit_t2i_v1_stack':
        from libs.u2vit_t2i import UViTStack  # with 4 blocks each

        # Initialize the NestedUViT model with the checkpoint path and additional arguments
        checkpoint_path = "/scratch/laks/U-ViT/workdir/medimgs/depth=16/ckpts/290000.ckpt/nnet_ema.pth"  # Update with the actual path
        model = UViTStack(checkpoint_path, **kwargs)  # Initialize UViTStack model with loaded checkpoint

        print("UViT weights loaded successfully into UViTStack.")
        return model

    elif name == 'm_uvit': # replacing attention blocks with mamba blocks
        from libs.m_uvit import UViT
        return UViT(**kwargs)        
    elif name == 'umvit': # replacing attention blocks and mamba blocks alternatively
        from libs.umvit import UViT
        return UViT(**kwargs)
    elif name == 'umvit_2': # drop path, mixer_cls and create_block classes of mamba
        from libs.umvit_2 import UViT
        return UViT(**kwargs)
    elif name == 'umzvit': # zigma
        from libs.umzvit import UViT
        return UViT(**kwargs)                        
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')
    print(nnet.eval)



### Compute flops

    # from deepspeed.profiling.flops_profiler import get_model_profile

    # macs, params = get_model_profile(wrapped_nnet, input_shape=(64, 4, 32, 32))

    # print(f"MACs: {macs}, Parameters: {params}")

#####

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
