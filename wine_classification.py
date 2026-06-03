from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from datetime import datetime

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    # 비교용 기본 모델을 준비한다.
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("model", RandomForestClassifier(random_state=seed)),
            ]
        ),
    }


def tune_random_forest(x_train, y_train, cv, seed: int = 42):
    # 랜덤 포레스트의 주요 하이퍼파라미터를 탐색한다.
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=seed),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Wine 분류 실험 스크립트")
    parser.add_argument("--seed", type=int, default=42, help="실험 시드 (기본: 42)")
    parser.add_argument("--out", type=str, default=".", help="결과 출력 디렉터리")
    parser.add_argument("--no-save-model", action="store_true", help="모델 파일 저장을 하지 않음")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    args = parse_args()
    setup_logging()
    seed = args.seed
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("데이터 로드 시작")
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=seed,
        stratify=dataset.target,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = build_models(seed)
    results = []
    fitted_models = {}

    start_time = time.time()
    logging.info("기본 모델 학습 및 평가 시작")
    for name, model in models.items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        logging.info(f"모델: {name} 정확도={accuracy:.4f} CV={cv_scores.mean():.4f}")
        results.append(
            {
                "name": name,
                "accuracy": float(accuracy),
                "cv_accuracy": float(cv_scores.mean()),
                "confusion": confusion_matrix(y_test, predictions).tolist(),
            }
        )
        fitted_models[name] = model

    logging.info("랜덤 포레스트 튜닝 시작")
    tuned_search = tune_random_forest(x_train, y_train, cv, seed=seed)
    tuned_model = tuned_search.best_estimator_
    tuned_predictions = tuned_model.predict(x_test)
    tuned_accuracy = accuracy_score(y_test, tuned_predictions)

    logging.info(f"튜닝 결과: RandomForest 정확도={tuned_accuracy:.4f} CVbest={tuned_search.best_score_:.4f}")
    results.append(
        {
            "name": "RandomForest(GridSearchCV)",
            "accuracy": float(tuned_accuracy),
            "cv_accuracy": float(tuned_search.best_score_),
            "confusion": confusion_matrix(y_test, tuned_predictions).tolist(),
            "best_params": tuned_search.best_params_,
        }
    )
    fitted_models["RandomForest(GridSearchCV)"] = tuned_model

    best_result = max(results, key=lambda item: item["accuracy"])
    logging.info(f"최종 선택 모델: {best_result['name']} 정확도={best_result['accuracy']:.4f}")

    # 선택된 모델 저장
    if not args.no_save_model:
        best_model = fitted_models[best_result["name"]]
        model_path = out_dir / "best_model.pkl"
        with model_path.open("wb") as model_file:
            pickle.dump(best_model, model_file)
        logging.info(f"모델 저장: {model_path}")

    # 결과 저장
    summary = {
        "best_model": best_result["name"],
        "accuracy": best_result["accuracy"],
        "cv_accuracy": best_result["cv_accuracy"],
        "results": results,
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    json_path = out_dir / "wine_results.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"요약 저장: {json_path}")

    text_lines = ["OSS ML Wine 결과", f"최종 선택 모델: {best_result['name']}", f"정확도: {best_result['accuracy']:.4f}", f"교차검증 정확도: {best_result['cv_accuracy']:.4f}"]
    text_path = out_dir / "wine_results.txt"
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    logging.info(f"텍스트 결과 저장: {text_path}")

    elapsed = time.time() - start_time
    logging.info(f"전체 소요 시간: {elapsed:.2f}초")


if __name__ == "__main__":
    main()
