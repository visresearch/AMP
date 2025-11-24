
import torch
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


    def synchronize_between_processes(self):  
        pack = torch.tensor([self.sum, self.count], device='cuda')
        dist.barrier()
        dist.all_reduce(pack)
        self.sum, self.count = pack.tolist()

    
    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):
        fmtstr = '{} {' + self.fmt + '} ({' + self.fmt + '})'
        return fmtstr.format(self.name, self.val, self.avg)


def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    if arr[left] >= target:
        return left

    while right - left > 1:
        mid = (left + right) // 2
        
        if arr[mid] >= target:
            right = mid
        else:
            left = mid

    if arr[right] >= target:
        result = right

    return result


import os
import shutil

def copy_files(src_dir, dst_dir, exclude_file_list):
    fnames = os.listdir(src_dir)

    os.makedirs(dst_dir, exist_ok=True)

    for f in fnames:
        if f not in exclude_file_list:
            src = os.path.join(src_dir, f)
            if os.path.isdir(src):
                dst = os.path.join(dst_dir, f)
                print(f'copy {src} to {dst}')
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                print(f'copy {src} to {dst_dir}')
                shutil.copy(src, dst_dir)
            else:
                ValueError(f'{src} can not be copied')

    return

def remove_key_prefix(state_dict, del_key):
    # del_key = 'model.'
    keys = list(state_dict.keys())
    for k in keys:
        if del_key in k:
            state_dict[k[len(del_key):]] = state_dict[k]
            del state_dict[k]
    return state_dict