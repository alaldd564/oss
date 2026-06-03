from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models() -> dict[str, Pipeline]:
    # 비교용 기본 모델을 준비한다.
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("model", RandomForestClassifier(random_state=42)),
            ]
        ),
    }


def tune_random_forest(x_train, y_train, cv):
    # 랜덤 포레스트의 주요 하이퍼파라미터를 탐색한다.
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [4, 6, None],
            "min_samples_split": [2, 4, 6],
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    lines: list[str] = []
    results = []

    # 각 모델을 학습하고 테스트 성능과 교차검증 성능을 함께 확인한다.
    for name, model in models.items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"모델: {name}")
        print(f"정확도: {accuracy:.4f}")
        print(f"교차검증 평균: {cv_scores.mean():.4f}")
        print("예측 결과 앞 5개:", predictions[:5])
        print(classification_report(y_test, predictions, target_names=dataset.target_names))
        print("-")

        results.append(
            {
                "name": name,
                "accuracy": float(accuracy),
                "cv_accuracy": float(cv_scores.mean()),
                "confusion": confusion_matrix(y_test, predictions).tolist(),
            }
        )

    # 랜덤 포레스트를 추가로 튜닝해서 더 좋은 조합을 찾는다.
    tuned_search = tune_random_forest(x_train, y_train, cv)
    tuned_model = tuned_search.best_estimator_
    tuned_predictions = tuned_model.predict(x_test)
    tuned_accuracy = accuracy_score(y_test, tuned_predictions)

    print("모델: RandomForest(GridSearchCV)")
    print(f"정확도: {tuned_accuracy:.4f}")
    print(f"교차검증 최고점: {tuned_search.best_score_:.4f}")
    print(f"최적 파라미터: {tuned_search.best_params_}")
    print(classification_report(y_test, tuned_predictions, target_names=dataset.target_names))

    results.append(
        {
            "name": "RandomForest(GridSearchCV)",
            "accuracy": float(tuned_accuracy),
            "cv_accuracy": float(tuned_search.best_score_),
            "confusion": confusion_matrix(y_test, tuned_predictions).tolist(),
            "best_params": tuned_search.best_params_,
        }
    )

    best_result = max(results, key=lambda item: item["accuracy"])
    print("최종 선택 모델:", best_result["name"])
    print(f"최고 정확도: {best_result['accuracy']:.4f}")

    # 결과를 파일로 저장해 제출 자료로 활용할 수 있게 한다.
    summary = {
        "best_model": best_result["name"],
        "accuracy": best_result["accuracy"],
        "cv_accuracy": best_result["cv_accuracy"],
        "results": results,
    }
    Path("wine_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines.append("OSS ML Wine 결과")
    lines.append(f"최종 선택 모델: {best_result['name']}")
    lines.append(f"정확도: {best_result['accuracy']:.4f}")
    lines.append(f"교차검증 정확도: {best_result['cv_accuracy']:.4f}")
    Path("wine_results.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
