import ApplicationInit
import ApplicationRouter
from fastapi import FastAPI

def create_app(config_path: str) -> FastAPI:
    return Builder.app_factory(config_path)


def get_app_instance() -> FastAPI:
    return Builder.get_app_instance()


class Builder:
    APP_INSTANCE: FastAPI = None
    CREATED: bool = False

    @staticmethod
    @ApplicationRouter.Wire
    def _initialize(config_path: str) -> ApplicationInit.ClassificationApp:
        return ApplicationInit.ClassificationApp(config_path)

    @staticmethod
    def app_factory(config_path: str) -> FastAPI:
        application = Builder._initialize(config_path)
        Builder.APP_INSTANCE = application.app
        Builder.CREATED = True
        return Builder.APP_INSTANCE

    @staticmethod
    def get_app_instance() -> FastAPI:
        if Builder.CREATED:
            return Builder.APP_INSTANCE
        else:
            Builder.app_factory()

