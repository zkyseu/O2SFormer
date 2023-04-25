import argparse
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

import dnlane
from dnlane.apis.inference import show_result_pyplot,inference_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint path')
    parser.add_argument('--img_path', help='image_path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device='cuda:0'
    config = mmcv.Config.fromfile(args.config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    # Use the detector to do inference
    img = args.img_path
    result = inference_detector(model, img)

    # Let's plot the result
    show_result_pyplot(model, img, result)