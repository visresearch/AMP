
from typing import Callable, Dict, Any
from collections import OrderedDict

from torch import nn

class MEOptimizer:
    def __init__(self, model:nn.Module, opt_cls:Callable, **kwargs):
        self.model = model

        self.optimizers = OrderedDict()
        self.lr = kwargs['lr']

        for param in model.parameters():
            if param.requires_grad:
                self.optimizers[param] = \
                    opt_cls(params=[param], **kwargs)
                
                param.register_post_accumulate_grad_hook(
                    self.release_param_grads
                )
        
        # --------- name to optimizer dict ---------
        self.name2opt = OrderedDict()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.name2opt[name] = self.optimizers[param]

    def release_param_grads(self, param):
        self.optimizers[param].step()
        self.optimizers[param].zero_grad(set_to_none=True)
    
    def set_lr_wd_for_optimizers(self, lr=None, wd=None):
        self.lr = lr
        for name, optimizer in self.name2opt.items():
            for param_group in optimizer.param_groups:
                if lr is not None: 
                    param_group["lr"] = lr
                if wd is not None: 
                    param_group["weight_decay"] = wd

    def state_dict(self) -> Dict[str, Any]:
        state_dict = OrderedDict()

        for name, opt in self.name2opt:
            state_dict[name] = opt.state_dict()

        return state_dict

    def load_state_dict(self, state_dict:Dict[str, Any]):
        params_covered = set()

        for param_name in state_dict.keys():
            if param_name not in self.name2opt:
                raise RuntimeError(
                    f"Trying to load optimizer state for unexpected param {param_name}"
                )

            self.name2opt[param_name].load_state_dict(state_dict[param_name])

            params_covered.add(param_name)

        # Ensure all params have been loaded into, report missing params
        missing_params = set(self.name2opt.keys()) - params_covered

        if missing_params:
            raise RuntimeError(
                f"Expected to load optimizer state for params {missing_params}!"
            )
    
