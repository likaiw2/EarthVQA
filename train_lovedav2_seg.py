import ever as er
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import json
from module import viz
# from data.lovedav2 import COLOR_MAP
from data.cityscapes import COLOR_MAP
import argparse
er.registry.register_all()

train_class = 19


def evaluate_cls_fn(self, test_dataloader, config=None, test_number=5):
    self.model.eval()
    seg_metric = er.metric.PixelMetric(train_class, logdir=self._model_dir, logger=self.logger)
    vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = viz.VisualizeSegmm(vis_dir, palette)


    with torch.no_grad():
        count = 0
        for img, ret in tqdm(test_dataloader,total=test_number):
            pred_seg = self.model(img, ret)
            if isinstance(pred_seg, tuple):
                pred_seg = pred_seg[0]
            seg_gt = ret['mask'].cpu().numpy()
            # calculate segmentation accuracy
            pred_seg = pred_seg.argmax(dim=1).cpu().numpy()
            valid_inds = seg_gt != -1
            
            # print("seg_gt shape:", seg_gt.shape)
            # print("pred_seg shape:", pred_seg.shape)
            # print("valid_inds shape:", valid_inds.shape)
            
            seg_metric.forward(seg_gt[valid_inds], pred_seg[valid_inds])

            for pred_seg_i, imagen_i in zip(pred_seg, ret['imagen']):
                viz_op(pred_seg_i, imagen_i.replace('jpg', 'png'))
            count += 1
            
            if count > test_number:
                break

    # seg_metric.summary_iou()
    final_metric = seg_metric.summary_all()
    torch.cuda.empty_cache()
    return final_metric



def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)



def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



if __name__ == '__main__':
    seed_torch(42)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', default='sfpnr50', type=str, help='path to config file')
    parser.add_argument('--model_dir', default='./log/sfpnr50', type=str, help='path to model directory')
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument('--trainer', default='th_ddp', type=str, help='type of trainer')
    parser.add_argument('--find_unused_parameters', action='store_true', help='whether to find unused parameters')
    parser.add_argument('--mixed_precision', default='fp32', type=str, help='datatype', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--use_wandb', action='store_true', help='whether to use wandb for logging')
    parser.add_argument('--project', default=None, type=str, help='Project name for init wandb')
    # command line options
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    
    trainer = er.trainer.get_trainer('th_ddp',parser)()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
    
