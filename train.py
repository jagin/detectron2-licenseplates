import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch

from licenseplates.dataset import register_licenseplates_voc
from licenseplates.config import setup_cfg
from licenseplates.trainer import Trainer


def main(args):
    # Register licenseplates dataset
    register_licenseplates_voc("licenseplates_train", "datasets/licenseplates", "train")
    register_licenseplates_voc("licenseplates_test", "datasets/licenseplates", "test")

    # Setup model configuration
    cfg = setup_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    # Run training process
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
