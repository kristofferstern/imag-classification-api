from fastapi import FastAPI, File, UploadFile

from image_classification.classifier import ImageClassifier

app = FastAPI()

classifier_config_path = "config.yaml"

# classifier = ImageClassifier.init_from_config_url(predictor_config_path)
classifier = ImageClassifier.init_from_config_path(classifier_config_path)

@app.post("/classifyfile/")
def create_upload_file(file: UploadFile = File(...)):
    return classifier.classify_from_file(file.file)