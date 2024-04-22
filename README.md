# T-Lymphocyte Detection on Immunohistochemistry Slides

This script is designed for performing object detection inference on CD3-stained immunohistochemistry (IHC) samples using the RetinaNet model architecture. Below is a brief overview of the script and its functionalities.

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

1. Inference: Given a directory containing slide images (--slide_dir), the script performs object detection inference using the RetinaNet model. It processes each slide image at a specified resolution level (--level) and patch size (--patch_size). The inference results are saved as CSV files containing bounding box coordinates and class labels.

2. Visualization: Optionally, you can visualize the detected objects overlaid on the slide images by setting the visualize flag to True. This generates visualization images stored alongside the CSV files.

## Usage
The script can be run from the command line with the following command:

```
python inference_script.py --slide_dir <path_to_slide_directory> --model_path <model_name> --level <resolution_level> --patch_size <patch_size>
```

- *slide_dir*: Path to the directory containing slide images.
- *model_path*: Name of the pre-trained model to use ('hnscc', 'nsclc', 'tnbc', 'gc', or 'pan_tumor').
- *level*: Resolution level (0 for the original resolution, higher levels for downsampled resolutions).
- *patch_size*: Size of the patches to be processed by the model (default is 256 x 256 pixels).

## Example Usage
```
python inference_script.py --slide_dir /path/to/slides --model_path all --level 0 --patch_size 256
```

This command performs inference using all available pre-trained models on the slide images in the specified directory at the original resolution (level 0) with a patch size of 256 x 256 pixels.

## Note

- The script assumes the availability of pre-trained RetinaNet models for different types of tissues (HNSCC, NSCLC, TNBC, GC). These models should be stored in a directory named ckpts.
- Ensure that the required dependencies are installed before running the script.

## Additional Notes

- For more details on the model training process, please refer to our puplished manuscript


- For dataset loading and preprocessing, please refer to the CD3Dataset class defined in the data.cd3_dataset module.

## Contributors

This script was developed by Frauke Wilm. For any inquiries or issues, please contact frauke.wilm@fau.de.