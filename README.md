# License plates detection model using Detectron2

For detailed description how to train your own detection model using a custom dataset and evaluate it read the Medium story:
* [Train license plates detection model using Detectron2](https://medium.com/deepvisionguru/train-license-plates-detection-model-using-detectron2-dd166154f604)

## Setup environment

This project is using [Conda](https://conda.io) for project environment management.

Setup the project environment:

    $ conda env create -f environment.yml
    $ conda activate detectron2-licenseplates
    
or update the environment if you `git pull` the repo previously:

    $ conda env update -f environment.yml
    
## Training

To launch end-to-end license plates detection training with Faster R-CNN ResNet-50 backbone on 2 GPUs,
one should execute:

    $ python train.py --config-file configs/lp_faster_rcnn_R_50_FPN_3x.yaml --num-gpus 2
    
To train the model with RetinaNet ResNet-50 backbone run:

    $ python train.py --config-file configs/lp_retinanet_R_50_FPN_3x.yaml --num-gpus 2

## Evaluation

Model evaluation is done at the and of the training but you can run it alone:

    $ python train.py --config-file configs/lp_faster_rcnn_R_50_FPN_3x.yaml --eval-only MODEL.WEIGHTS output/model_final.pth
    
or

    $ python train.py --config-file configs/lp_retinanet_R_50_FPN_3x.yaml --eval-only MODEL.WEIGHTS output/model_final.pth

## Prediction

To execute prediction on some sample data from test dataset with Faster R-CNN ResNet-50 backbone (which is default), run:

    $ python predict.py --config-file configs/lp_faster_rcnn_R_50_FPN_3x.yaml MODEL.WEIGHTS output/model_final.pth
    
or

    $ python predict.py --config-file configs/lp_retinanet_R_50_FPN_3x.yaml MODEL.WEIGHTS output/model_final.pth
    
to run prediction on RetinaNet

## Resources and Credits

* [Detectron2](https://github.com/facebookresearch/detectron2)
* [How to embed Detectron2 in your computer vision project](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)

## License

[MIT License](LICENSE)