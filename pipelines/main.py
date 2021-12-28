from pathlib import Path
import logging
import warnings
warnings.filterwarnings(action='ignore')

from train import train_model
from inference import make_predictions

logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = Path('./')
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'

data_filepath = '..' / DATA_DIR / 'loan_eligibility.xlsx'

response = train_model(data_filepath, MODELS_DIR)
print(response)
response = make_predictions(data_filepath, MODELS_DIR)
print(response)
