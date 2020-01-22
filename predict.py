import random
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

from licenseplates.dataset import register_licenseplates_voc
from licenseplates.config import setup_cfg


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Licenseplates prediction")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)

    # Detectron settings
    ap.add_argument("--config-file",
                    required=True,
                    help="path to config file")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="minimum score for instance predictions to be shown (default: 0.5)")
    ap.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                    help="modify model config options using the command-line")

    return ap.parse_args()


def main(args):
    if args.confidence_threshold is not None:
        # Set score_threshold for builtin models
        args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))
        args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))

    dataset_name = "licenseplates_test"
    register_licenseplates_voc(dataset_name, "datasets/licenseplates", "test")
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, args.samples):
        img = cv2.imread(d["file_name"])
        prediction = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=MetadataCatalog.get(dataset_name),
                                scale=args.scale)
        vis = visualizer.draw_instance_predictions(prediction["instances"].to("cpu"))
        cv2.imshow(dataset_name, vis.get_image()[:, :, ::-1])

        # Exit? Press ESC
        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)

    main(args)
