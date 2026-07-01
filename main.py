import os

from ultralytics import YOLO

INPUT_FOLDER = "./Images"
MODEL_FOLDER = "./Ai_Models"
RESULT_FOLDER = "./Results"


def try_Ai(model: str, input: str, output_folder: str = "./Results"):
    """
    model is path to the best.pt
    input is path to the image
    output_folder is path where the picture will be stored
    """
    yolo_model = YOLO(model)
    results = yolo_model(input)
    show_results(results)
    os.makedirs(output_folder, exist_ok=True)
    results[0].save(os.path.join(output_folder, os.path.basename(input)))


def show_results(results):
    results[0].show()


def main():
    for model_file in os.listdir(MODEL_FOLDER):
        model_path = os.path.join(MODEL_FOLDER, model_file)
        for image_file in os.listdir(INPUT_FOLDER):
            image_path = os.path.join(INPUT_FOLDER, image_file)
            try_Ai(model_path, image_path, RESULT_FOLDER)


if __name__ == "__main__":
    main()
