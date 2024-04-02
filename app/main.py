import uvicorn
from ApplicationBuilder import create_app

config_path = "config.yaml"
app = create_app(config_path)

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')