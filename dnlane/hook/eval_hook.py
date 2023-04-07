import os.path as osp
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

from ..apis import single_gpu_test,multi_gpu_test
from mmcv.runner import EvalHook,DistEvalHook

class LaneEvalHook(EvalHook):
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)

class LaneDistEvalHook(DistEvalHook):
    def __init__(self,dataloader,**kwargs):
        if 'test_fn' not in kwargs.keys():
            test_fn = multi_gpu_test
            kwargs['test_fn'] = test_fn
        super().__init__(dataloader,**kwargs)
