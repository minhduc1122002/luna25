from typing import Dict, Tuple

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
from scipy.special import logit
import joblib
import time
from processor import MalignancyProcessor


INPUT_PATH = Path("./test/input")   # data
OUTPUT_PATH = Path("./test/output") # output
RESOURCE_PATH = Path("./results") # checkpoint

def transform(input_image, point):
    """

    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points

    Returns
    -------
    tNumpyOrigin

    """
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )

def itk_image_to_numpy_image(input_image):
    """

    Parameters
    ----------
    input_image: SimpleITK image

    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format

    """

    numpyImage = SimpleITK.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header


class NoduleProcessor:
    def __init__(self, ct_image_file, nodule_locations, mode="3D", ensemble_weights=[0.4, 0.3, 0.3]):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        model_name: Name of the model to be used for prediction
        """
        self._image_file = ct_image_file
        self.nodule_locations = nodule_locations
        self.mode = mode
        self.ensemble_weights = ensemble_weights

        self.processor = MalignancyProcessor(mode=mode, suppress_logs=True, ensemble_weights=ensemble_weights)


    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> Dict:
        """

        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates in /input/nodule-locations.json

        Returns
        -------
        malignancy risk of the nodules provided in /input/nodule-locations.json
        """

        numpyImage, header = itk_image_to_numpy_image(input_image)

        malignancy_risks = []
        for i in range(len(coords)):
            self.processor.define_inputs(numpyImage, header, [coords[i]])
            malignancy_risk, logits = self.processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)

        malignancy_risks = np.array(malignancy_risks)


        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    def load_inputs(self):
        # load image
        print(f"Reading {self._image_file}")
        image = SimpleITK.ReadImage(str(self._image_file))

        self.annotationIDs = [str(self.nodule_locations["patientID"]) + '_' + str(self.nodule_locations["lesionID"]) + '_' + str(self.nodule_locations["studyDate"])]
        self.coords = np.array([[self.nodule_locations["coordZ"], self.nodule_locations["coordY"], self.nodule_locations["coordX"]]]) # [z, y, x] format

        return image, self.coords, self.annotationIDs

    def process(self):
        """
        Load CT scan(s) and nodule coordinates, predict malignancy risk and write the outputs
        Returns
        -------
        None
        """
        image, coords, annotationIDs = self.load_inputs()
        output = self.predict(image, coords)

        assert len(output) == len(annotationIDs), "Number of outputs should match number of inputs"
        # in this class, only 1 file for each request so the number of output is 1
        assert len(output) == 1
        return output[0]


def run(mode="3D", ensemble_weights=[0.4, 0.3, 0.3], image_path=None, metadata_path=None):
    start = time.time()

    # Read the inputs
    input_nodule_locations = read_json(metadata_path)
    input_chest_ct = image_path
    
    # Validate access to GPU
    _show_torch_cuda_info()
    
    # Run your algorithm here
    processor = NoduleProcessor(ct_image_file=input_chest_ct,
                                nodule_locations=input_nodule_locations,
                                mode=mode,
                                ensemble_weights=ensemble_weights)
    malignancy_risks = processor.process()
    pred_binary = (malignancy_risks > 0.5).astype(int)  # Apply threshold of 0.5
    ms = int((time.time() - start) * 1000)

    # Save your output
    result_data = {
        "seriesInstanceUID": input_nodule_locations["seriesInstanceUID"],
        "lesionID": input_nodule_locations["lesionID"],
        "probability": float(malignancy_risks),
        "predictionLabel": int(pred_binary),
        "processingTimeMs": ms
    }

    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # write to json file
    output_file = str(input_nodule_locations["seriesInstanceUID"]) + "_" + str(input_nodule_locations["lesionID"]) + ".json"
    write_json_file(
        location=OUTPUT_PATH / output_file,
        content=result_data,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {result_data}") 

    return result_data

def run_multiple(mode="3D", ensemble_weights=[0.4, 0.3, 0.3], image_folder=None, metadata_folder=None):

    tmp = Path(image_folder)
    result_list = []
    # Non-recursive: files only
    for p in tmp.iterdir():
        if p.is_file():
            metadata_path = metadata_folder + "/" + str(p.stem) + ".json"
            image_path = str(p)   

            start = time.time()

            # Read the inputs
            input_nodule_locations = read_json(metadata_path)
            input_chest_ct = image_path
    
            # Validate access to GPU
            _show_torch_cuda_info()
    
            # Run your algorithm here
            processor = NoduleProcessor(ct_image_file=input_chest_ct,
                                nodule_locations=input_nodule_locations,
                                mode=mode,
                                ensemble_weights=ensemble_weights)
            malignancy_risks = processor.process()
            pred_binary = (malignancy_risks > 0.5).astype(int)  # Apply threshold of 0.5
            ms = int((time.time() - start) * 1000)

            # Save your output
            result_data = {
                "seriesInstanceUID": input_nodule_locations["seriesInstanceUID"],
                "probability": float(malignancy_risks),
                "predictionLabel": int(pred_binary),
                "processingTimeMs": ms,
                "CoordX": input_nodule_locations["coordX"],
                "CoordY": input_nodule_locations["coordY"],
                "CoordZ": input_nodule_locations["coordZ"]
            }
            result_list.append(result_data)

    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # write to json file
    output_file = "batch_results.json"
    write_json_file(
        location=OUTPUT_PATH / output_file,
        content=result_list,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {result_list}") 

    return result_list

def read_json(path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json_file(location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_path(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )

    assert (
                len(input_files) == 1
            ), "Please upload only one .mha file per job for grand-challenge.org"
    
    result = input_files[0]

    return result


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.version.cuda}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    mode = "3D"
    #model_name = "LUNA25-pulse-baseline-Pulse3D-20251030"
    #model_name = "LUNA25-pulse-baseline-HyperPulse-20251127104650"        # HyperPulse
    # model_name = "/results/finetune-VJEPA2-20260320/fold_1/best_metric_model.pth"       # HyperPulse2     
    # image_path = "./test/input/images/1.2.840.113654.2.55.294281779470566559919697495520361195429.mha"
    # metadata_path = "./test/input/metadata/1.2.840.113654.2.55.294281779470566559919697495520361195429_2.json"     
    # raise SystemExit(run(mode= mode, model_name=model_name,
    #                     image_path=image_path, metadata_path=metadata_path))
    ensemble_weights = [0.5, 0.25, 0.25]
    image_folder = "./test/input/images/real_data"
    metadata_folder = "./test/input/metadata/real_data"
    raise SystemExit(run_multiple(mode= mode, ensemble_weights=ensemble_weights,
                         image_folder=image_folder, metadata_folder=metadata_folder))