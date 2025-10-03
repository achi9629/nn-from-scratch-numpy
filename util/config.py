from yacs.config import CfgNode as CN

config = CN()

config.seed = 0
config.epochs = 0
config.batch_size = 0
config.layer_dims = []

config.optimizer=CN()
config.optimizer.otype=''

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()