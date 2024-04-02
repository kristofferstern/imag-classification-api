from typing import Any
from fastapi import FastAPI, File, UploadFile

class Wire:
    def __init__(self, app_factory) -> None:
        self.app_factory = app_factory


    def __call__(self, config_path) -> FastAPI:
        app = self.app_factory(config_path)

        # Add decorations: API Routing

        @app.app.post("/classifyfile/")
        def create_upload_file(file: UploadFile = File(...)):
            return app.classifier.classify_from_file(file.file)

        return app
