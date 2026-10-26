
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

def test_prediction():
    model = joblib.load(MODEL_PATH)

     
    sample = [[5.1, 3.5, 1.4, 0.2]]

    prediction = model.predict(sample)

    print("Predicted Class:", prediction[0])

    assert prediction[0] in [0, 1, 2]

    print("Prediction Test Passed!")


if __name__ == "__main__":
    test_prediction()
