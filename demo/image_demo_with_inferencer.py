# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import MMSegInferencer
from mmengine.config import Config
import os

def remove_load_annotations(pipeline):
    for transform in pipeline:
        if transform.get('type') == 'TestTimeAug':
            tta_transforms = []
            for transforms in transform['transforms']:
                filtered_transforms = [
                    aug for aug in transforms
                    if aug.get('type') != 'LoadAnnotations'
                ]
                if filtered_transforms:
                    tta_transforms.append(filtered_transforms)
            transform['transforms'] = tta_transforms
    return pipeline

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='', help='Path to save result file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to display the drawn image.')
    parser.add_argument(
        '--dataset-name',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    args = parser.parse_args()

    model = args.model
    if args.tta:
        cfg = Config.fromfile(args.model)
        cfg.test_dataloader.dataset.pipeline = remove_load_annotations(
            cfg.tta_pipeline)
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model
        model = cfg

    # build the model from a config file and a checkpoint file
    mmseg_inferencer = MMSegInferencer(
        model,
        args.checkpoint,
        dataset_name=args.dataset_name,
        device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    # test a single image
    if os.path.isfile(args.img):
        mmseg_inferencer(
            args.img,
            show=args.show,
            out_dir=args.out_dir,
            opacity=args.opacity,
            with_labels=args.with_labels)
    elif os.path.isdir(args.img):
        image_path = args.img
        image_path = [os.path.join(image_path,i) for i in os.listdir(image_path)]
        for image_path_temp in image_path:
            mmseg_inferencer(
                image_path_temp,
                show=args.show,
                out_dir=args.out_dir,
                opacity=args.opacity,
                with_labels=args.with_labels)

if __name__ == '__main__':
    main()
