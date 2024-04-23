from object_detection_fastai.helper.object_detection_helper import *
from object_detection_fastai.models.RetinaNet import RetinaNet
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from data.cd3_dataset import CD3Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import openslide
import argparse
import torch
import cv2
import os

def visualize_detections(slide, detections, ds=1):
    # Visualize detections on the slide image
    slide_image = np.array(slide.get_thumbnail((slide.dimensions[0]//ds, slide.dimensions[1]//ds)))[:,:,:3]
    slide_image = cv2.cvtColor(slide_image, cv2.COLOR_RGB2BGR)
    colors = {
        'IMMUNE CELL': (0, 255, 0),  # Green
        'NON-TUMOR CELL': (255, 0, 0),  # Blue
        'TUMOR CELL': (0, 165, 255),  # Orange
    }

    for detection in detections:
        x1, y1, x2, y2, _, class_label = detection
        color = colors[class_label]
        cv2.rectangle(slide_image, (x1//ds, y1//ds), (x2//ds, y2//ds), color, 2)
    return slide_image

def load_model_and_anchors(model_path):
    print('Loading model')
    anchors = create_anchors(sizes=[(32, 32), (16, 16), (8, 8), (4, 4)], ratios=[0.5, 1, 2],scales=[0.5, 0.75, 1, 1.25, 1.5])
    encoder = create_body(resnet18(), pretrained=False, cut=-2)
    model = RetinaNet(encoder, n_classes=4, n_anchors=15, sizes=[32, 16, 8, 4], chs=128, final_bias=-4., n_conv=3)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'ckpts', f'{model_path}.pth'), map_location=torch.device('cpu'))['model'])
    return model, anchors

def inference(dataloader, model, anchors, device, patch_size, down_factor, detect_thresh=0.5, nms_thresh=0.5):
    classes = ['IMMUNE CELL', 'NON-TUMOR CELL', 'TUMOR CELL']
    class_pred_batch, bbox_pred_batch, x_coordinates, y_coordinates = [], [], [], []
    patch_counter = 0
    detections = []
    with torch.inference_mode():
        for patches,x,y in tqdm(dataloader):
            class_pred, bbox_pred, _ =  model(patches.to(device))
            class_pred_batch.extend (class_pred.cpu())
            bbox_pred_batch.extend(bbox_pred.cpu())
            x_coordinates.extend(x)
            y_coordinates.extend(y)
            patch_counter += len(patches)
        print(f'Ran inference for {patch_counter} patches.')

        final_bbox_pred, final_scores, final_class_pred = [], [], []
        print(f'Postprocessing predictions.')
        for clas_pred, bbox_pred, x, y in zip(class_pred_batch, bbox_pred_batch, x_coordinates, y_coordinates):
            # anchor matching and filter predictions with detection threshold 
            bbox_pred, scores, preds = process_output(clas_pred.cpu(), bbox_pred.cpu(), anchors, detect_thresh=detect_thresh)
            if bbox_pred is not None:
                # rescale detection boxes with patch_size
                t_sz = torch.Tensor([patch_size, patch_size])[None].float()
                bbox_pred = rescale_box(bbox_pred, t_sz)
                bbox_pred += torch.Tensor([y//down_factor, x//down_factor, 0, 0]).long()
                # apply non-maximum-supression per patch
                to_keep = nms(bbox_pred, scores, return_ids = True, nms_thresh = nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu() 
                final_bbox_pred.extend(bbox_pred)
                final_class_pred.extend(preds)
                final_scores.extend(scores)
        # Global non-maximum-supression
        keep_global = nms(torch.Tensor(np.array(final_bbox_pred)), torch.Tensor(final_scores), return_ids = True, nms_thresh = nms_thresh)
        final_bbox_pred = torch.Tensor(np.array(final_bbox_pred))[keep_global]
        final_class_pred = torch.Tensor(np.array(final_class_pred))[keep_global]
        final_scores = torch.Tensor(np.array(final_scores))[keep_global]

        # convert bbox (x_local, y_local, h, w) to (x1_global, y1_global, x2_global, y2_global)
        for box, pred, score in zip(final_bbox_pred, final_class_pred, final_scores):
            y_box, x_box = box[:2]
            h, w = box[2:4]

            x1 = int(x_box) * down_factor
            y1 = int(y_box) * down_factor
            x2 = x1 + int(w) * down_factor
            y2 = y1 + int(h) * down_factor

            detections.append([int(x1), int(y1), int(x2), int(y2), float(score), classes[int(pred)]])
    return detections

def process(slide_dir, model_path, level, patch_size, detect_thresh, visualize):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') 
    model, anchors = load_model_and_anchors(model_path)
    model = model.eval().to(device)
    os.makedirs(os.path.join(slide_dir, 'processing', model_path), exist_ok=True)

    # Iterate through slides in folder.
    print(f'Running inference with {model_path}.pth model on all slides in folder {slide_dir} using a patch size of {patch_size}x{patch_size} at resolution level {level}')
    for file in os.listdir(slide_dir):
        print(f'Slide {file}')
        slide = openslide.open_slide(os.path.join(slide_dir, file))
        ds = CD3Dataset(slide, level, patch_size)
        dl = DataLoader(ds, num_workers=0, batch_size=8)
        detections = inference(dl, model, anchors, device, patch_size, slide.level_downsamples[level], detect_thresh)
        detection_df = pd.DataFrame(detections, columns=['x1','y1','x2','y2','score','class'])
        result_path = os.path.join(slide_dir, 'processing', model_path, f"{file.split('.')[0]}.csv")
        detection_df.to_csv(result_path, index=False)
        print('Stored results at', result_path)
        # Visualize results 
        if visualize:
            slide_image_with_detections = visualize_detections(slide, detections)
            cv2.imwrite(result_path.replace('.csv', '.png'), slide_image_with_detections)
            print('Stored visualization of predictions at', result_path.replace('.csv', '.png')) 

def main():
    parser = argparse.ArgumentParser(description='Inference for T-lyphocyte detection on CD3-stained IHC samples')
    parser.add_argument('--slide_dir', type=str, help='Slide directory')
    parser.add_argument('--model_path', type=str, help='Model weights', default='pan_tumor', choices=['hnscc', 'nsclc', 'tnbc', 'gc', 'pan_tumor'])
    parser.add_argument('--level', type=int, help='Resolution level (models were trained on level 0, i.e. 0.25 um/pixel)', default=0)
    parser.add_argument('--patch_size', type=int, help='Patch size (models were trained on 256 x 256 pixels)', default=256)
    parser.add_argument('--detect_thresh', type=int, help='Confidence threshold for detections. Lower threshold increases recall, higher threshold increases specificity.', default=0.5)
    args = parser.parse_args()

    # Call inference function with parsed arguments
    process(slide_dir=args.slide_dir, model_path=args.model_path, level=args.level, patch_size=args.patch_size, detect_thresh=args.detect_thresh, visualize=False)

if __name__ == "__main__":
    main()