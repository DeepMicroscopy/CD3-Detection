# T-Lymphocyte Detection on Immunohistochemistry Slides

This script is designed for performing object detection inference on CD3-stained immunohistochemistry (IHC) samples using the RetinaNet model architecture. Below is a brief overview of the script and its functionalities.

![Detection Example.](example.png)

## Setup

**Requirements**: 
  - Python (>=3.6)
  - PyTorch
  - torchvision
  - fastai
  - pandas
  - tqdm
  - openslide
  - cv2

## Functionality

The script provides the following functionalities:

1. Inference: Given a directory containing slide images (--slide_dir), the script performs object detection inference using the RetinaNet model. It processes each slide image at a specified resolution level (--level) and patch size (--patch_size). The sensitivity of the detections can be adapted with the the detection threshold (--detect_thresh). The inference results are saved as CSV files containing bounding box coordinates and class labels.

2. Visualization: Optionally, you can visualize the detected objects overlaid on the slide images by setting the visualize flag to True. This generates visualization images stored alongside the CSV files.

## Usage
The script can be run from the command line with the following command:

```
python inference_script.py --slide_dir <path_to_slide_directory> --level <resolution_level> --patch_size <patch_size> --detect_thresh <detect_thresh> --visualize <visualize>
```

- *slide_dir*: Path to the directory containing slide images.
- *level*: Resolution level (0 for the original resolution, higher levels for downsampled resolutions).
- *patch_size*: Size of the patches to be processed by the model (default is 256 x 256 pixels).
- *detect_thresh*: Confidence threshold for detection. Lower threshold increases recall, higher threshold increases specificity (default is 0.5).
- *visualize*: Flag for exporting visual detection results. Default is FALSE. 

## Example Usage
```
python inference_script.py --slide_dir /path/to/slides --level 0 --patch_size 256 --detect_thresh 0.5 --visualize True
```

This command performs inference using all available pre-trained models on the slide images in the specified directory at the original resolution (level 0) with a patch size of 256 x 256 pixels using a detection threshold of 0.5. The comman will create a detection .csv and result .png for each slide. 

## Note

- The script assumes the availability of pre-trained RetinaNet models. These models should be stored in a directory named ckpts.
- Ensure that the required dependencies are installed before running the script.

## Additional Notes

- For more details on the model training process, please refer to our puplished manuscript:
> Wilm, Frauke, et al. "Pan-tumor T-lymphocyte detection using deep neural networks: Recommendations for transfer learning in immunohistochemistry." Journal of Pathology Informatics 14 (2023): 100301.

- The dataset and annotations used for training the detection models can be downloaded from [Zenodo](https://zenodo.org/records/7500843). 
- For dataset loading and preprocessing, please refer to the CD3Dataset class defined in the [cd3_dataset](data/cd3_dataset.py) module.

## Contributors

This script was developed by Frauke Wilm. For any inquiries or issues, please contact frauke.wilm@fau.de.