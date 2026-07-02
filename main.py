import os

import ultralytics
from ultralytics import YOLO

INPUT_FOLDER: str = "./Images"
MODEL_FOLDER: str = "./Ai_Models"
RESULT_FOLDER: str = "./Results"


def try_Ai(model: str, input: str, output_folder: str = "./Results"):
    """
    model is path to the best.pt
    input is path to the image
    output_folder is path where the picture will be stored
    """
    yolo_model = YOLO(model)
    filename, _ = os.path.splitext(os.path.basename(input))
    save_path = os.path.join(output_folder, f"{filename}.txt")
    results: list[ultralytics.engine.results.Results] = yolo_model(input, save_txt=None)
    with open(save_path, "w") as file:
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            cls = int(results[0].boxes.cls[idx].item())
            file.write(
                f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n"
            )

    if not results or len(results[0].boxes) <= 0:
        print(f"No detections for {input}")
        return
    show_results(results)
    os.makedirs(output_folder, exist_ok=True)
    results[0].save(os.path.join(output_folder, os.path.basename(input)))


def show_results(results):
    results[0].show()


def main():
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
