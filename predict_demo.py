from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    # 저장한 모델 파일을 불러온다.
    model_path = Path("best_model.pkl")
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    # 같은 데이터셋을 다시 불러와 예측 결과를 확인한다.
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    # 예측 결과와 정확도를 출력한다.
    print("모델 파일:", model_path.name)
    print("예측 앞 5개:", predictions[:5])
    print(f"데모 정확도: {accuracy:.4f}")


if __name__ == "__main__":
    main()
