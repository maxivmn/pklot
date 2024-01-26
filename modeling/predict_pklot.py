import sys
from fastapi import FastAPI
from sklearn.metrics import mean_squared_error
import warnings
import mlflow
from mlflow.sklearn import load_model

warnings.filterwarnings("ignore")

from .feature_eng_pklot import *

app = FastAPI()

def ensemble_models(images, model_path):
    pass
    
    
# Helper functions
def load_models(path):
    pass

def visualize(image, xml_path, model_path):
    pass