import os

import ultralytics
from ultralytics import YOLO

INPUT_FOLDER: str = "./Images"
MODEL_FOLDER: str = "./Ai_Models"
RESULT_FOLDER: str = "./Results"


def try_Ai(model: str, input: str, output_folder: str = "./Results"):
    """
    Run YOLO object detection on an image and save the results.

    Runs inference on the given image, writes the detected bounding boxes
    (in normalized xywh format, one per line, prefixed by class id) to a
    .txt file named after the input image, and saves an annotated copy of
    the image if any detections were found.

    Args:
        model (str): Path to the YOLO model weights (e.g. "best.pt").
        input (str): Path to the input image.
        output_folder (str): Folder where the label .txt file and the
            annotated image will be saved. Defaults to "./Results".
    """
    yolo_model: YOLO = YOLO(model)
    classes_txt = make_classes_file(yolo_model.names, output_folder)
    filename, _ = os.path.splitext(os.path.basename(input))
    save_path: str = os.path.join(output_folder, f"labels/{filename}.txt")
    results: list[ultralytics.engine.results.Results] = yolo_model(input, save_txt=None)
    with open(save_path, "a") as file:
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            cls = int(results[0].boxes.cls[idx].item())
            file.write(
                f"{get_correct_cls(yolo_model.names, cls, classes_txt)} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n"
            )

    if not results or len(results[0].boxes) <= 0:
        print(f"No detections for {input}")
        return

    results[0].save(os.path.join(output_folder, "images/", os.path.basename(input)))


def make_classes_file(names: dict, output_folder: str) -> str:

    save_path: str = os.path.join(output_folder, "classes.txt")

    if os.path.isfile(save_path):
        with open(save_path, "rt") as myfile:
            existing = {line.strip() for line in myfile}

        # remove entries from names whose value is already in the file
        names = {idx: name for idx, name in names.items() if name not in existing}
    with open(save_path, "a") as file:
        for idx in sorted(names.keys()):
            file.write(f"{names[idx]}\n")
    return save_path


def get_correct_cls(names: dict, cls: int, classes_txt: str) -> int:
    """"""

    with open(classes_txt, "rt") as myfile:
        for i, myline in enumerate(myfile):
            if myline.strip() == str(names[cls]).strip():
                print(i)
                return i
    raise ValueError("Something went wrong")


def make_folder_structer(output_folder: str, delete_files: bool = True):
    """Create the output folder structure for results.

    Creates the given output folder (if it doesn't already exist) along
    with two subfolders inside it: "labels" and "images".

    Args:
        output_folder (str): Path to the base output folder."""
    labels_path: str = os.path.join(output_folder, "labels")
    if delete_files and os.path.isfile(os.path.join(output_folder, "classes.txt")):
        for images_file in os.listdir(labels_path):
            os.remove(os.path.join(labels_path, images_file))
        os.remove(os.path.join(output_folder, "classes.txt"))
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)


def main():

    make_folder_structer(output_folder=RESULT_FOLDER)
    for model_file in os.listdir(MODEL_FOLDER):
        _, end = os.path.splitext(os.path.basename(model_file))
        if end != ".pt":
            continue
        model_path = os.path.join(MODEL_FOLDER, model_file)
        for image_file in os.listdir(INPUT_FOLDER):
            _, end = os.path.splitext(os.path.basename(image_file))
            if end != ".jpg" and end != ".png":
                continue
            image_path = os.path.join(INPUT_FOLDER, image_file)
            try_Ai(model_path, image_path, RESULT_FOLDER)


if __name__ == "__main__":
    main()
