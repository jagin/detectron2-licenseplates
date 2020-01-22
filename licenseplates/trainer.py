from detectron2.engine import DefaultTrainer

from .evaluator import VOCDetectionEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return VOCDetectionEvaluator(dataset_name)
