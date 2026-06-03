from __future__ import annotations

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # 와인 데이터셋을 불러온다.
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    # 선형 분류기와 트리 기반 분류기를 비교한다.
    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    best_name = None
    best_accuracy = -1.0

    for name, model in models.items():
        # 모델을 학습시키고 테스트셋 정확도를 계산한다.
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"모델: {name}")
        print(f"정확도: {accuracy:.4f}")
        print("예측 결과 앞 5개:", predictions[:5])
        print("-")

        if accuracy > best_accuracy:
            best_name = name
            best_accuracy = accuracy

    print("최종 선택 모델:", best_name)
    print(f"최고 정확도: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
