import torch

def gather_features(
        features,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_features, features)
        
        gathered_features[rank] = features

        all_features = torch.cat(gathered_features, dim=0)

    return all_features

