from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


SITE = {"name": "Ibri, Oman", "latitude": 23.23, "longitude": 56.52}


PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",   
    "CLRSKY_SFC_SW_DWN",   
    "TOA_SW_DWN",         
    "T2M",                 
    "RH2M",               
    "WS10M",               
    "PRECTOTCORR",        
]
START_YEAR, END_YEAR = 2015, 2025
FILL_VALUE = -999.0


TRAIN_END = "2022-12-31"
VALID_END = "2023-12-31"     


DAYLIGHT_TOA = 20.0           


PLANT_MW = 500.0
PLANT_FULL_OUTPUT_GHI = 1000.0  

SEED = 42
