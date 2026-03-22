# Finetuning Hiera/VideoMAE/VJEPA2 on LUNA25 3D Patches

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Code License](https://img.shields.io/badge/code_license-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

This work, participating in the [LUNA25 Challenge](https://luna25.grand-challenge.org/), finetuned Hiera for lung nodule malignancy risk estimation.

## Usage
### Set Up
```
conda create -y -n venv2025 python==3.11
conda activate venv2025
pip install -r requirements.txt
```

### Get Data
```
wget https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.001
wget https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.002
wget https://zenodo.org/records/14673658/files/LUNA25_Public_Training_Development_Data.csv

cat luna25_nodule_blocks.zip.001 luna25_nodule_blocks.zip.002 > luna25_nodule_blocks.zip
apt update && apt -y install p7zip-full
7z x luna25_nodule_blocks.zip
rm luna25_nodule_blocks.zip.001 luna25_nodule_blocks.zip.002
```

### Perform Training

```
python split_data.py
python finetune.py --k_fold
```
When training using all data, set `kind="all-data"` in `experiment_config.py` then run
```
python finetune.py --all_data
```

After training the checkpoints will be saved in results folder

To run Ensemble do, Remember to change file path:
```
do_ensemble.sh
```
### Inference with Real Data

1. Preparing the test data:
- Assume that you have the input in the DICOM format file. Add it in 'data' folder like this:
```
data
  |- metadata (luna data)
  |- HA251209.177_CT (real data)
	|- 3-1.25mm NHU MO PHOI
	|- 4-1.25mm Trung That
```
- Run data preprocessing. This will create the .mha image file and .json metadata file in the 'test' folder. :
```bash
python MedSAM2/data_preprocessing.py
```
- Make sure that you have unique seriesInstanceUID in DICOM files in each folder. If you have the same seriesInstanceUID in multiple folders, our code will only keep one. For example, our folders  '123.255088989375905.1851013063013701 (3)' and '123.255088989375905.1851013063013701 (4)' in 'data' have the same seriesInstanceUID, meaning that they are duplicated. So our code will remove duplication. So if you receive predictions that do not match with your number of testcase, please check your seriesInstanceUID

2. **Configure the inference script**

Open the `inference_real.py` script and configure:
- `INPUT_PATH`: Path to the input data (CT, nodule locations and clinical information). Keep as `Path("./test/input")` for Grand-Challenge.
- `RESOUCE_PATH`: Path to resources (e.g., pretrained models weights) in the container. Defaults to `./results` directory (see Dockerfile)
- `OUTPUT_PATH`: Path to store the output in your local directory. Keep as `Path("./test/output")` for Grand-Challenge.
- **Inputs for the `run()` function**:
    - `mode`: Match this to the mode used during training (2D or 3D).
    - `model_name`: Specify the experiment_name matching the training configuration (corresponding to experiment_name directory that contains the model weights in `/results`). You can install our checkpoint [here](https://drive.google.com/drive/folders/13S_gRJN9q8vOd4HQPPdhOs5Y0XftmhqH?usp=sharing).
- Run inference:
```bash
python inference_real.py
```

## License
This work's code is licensed under the [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).

## References

This work has been adapted from and is inspired by the following repositories:

- [luna25-baseline-public](https://github.com/DIAGNijmegen/luna25-baseline-public)
- [hiera](https://github.com/facebookresearch/hiera)
- [SlowFast](https://github.com/facebookresearch/SlowFast)
- [torchvision](https://github.com/pytorch/vision/tree/main/torchvision)