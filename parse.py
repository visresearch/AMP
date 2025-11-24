import argparse

def get_args_score():
    parser = argparse.ArgumentParser("Estimate importance score")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--frame", type=str)
    parser.add_argument("--layer_wise_prune", action='store_true')

    parser.set_defaults(
        config_file = "./configs/open_clip_coco.yaml",
        frame = 'open_clip',
        # layer_wise_prune = True,
        exclude_file_list = ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']
    )

    args = parser.parse_args()

    return args

def get_args_distill():
    parser = argparse.ArgumentParser("Distill for pruned model")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--frame", type=str)
    parser.add_argument("--print_freq", type=int)
    parser.add_argument("--save_freq", type=int)

    parser.set_defaults(
        config_file = "./configs/distill_clip.yaml",
        frame = 'open_clip',
        print_freq = 100,
        save_freq = 1,
        exclude_file_list = ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight']
    )

    args = parser.parse_args()

    return args