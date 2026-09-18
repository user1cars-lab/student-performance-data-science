import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import prepare_dataset
from src.modeling import train_models

if __name__ == "__main__":
    data = prepare_dataset()
    _, _, regression, classification = train_models(data)
    print("Regression results")
    print(regression.to_string(index=False))
    print("Classification results")
    print(classification.to_string(index=False))
