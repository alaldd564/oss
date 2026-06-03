from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="저장된 모델로 데모 예측 및 혼동행렬 시각화")
    parser.add_argument("--model", type=str, default=None, help="불러올 모델 파일 경로")
    parser.add_argument("--out", type=str, default=".", help="결과 출력 디렉터리")
    parser.add_argument("--seed", type=int, default=42, help="데이터 분할 시드")
    parser.add_argument("--save-plot", action="store_true", help="혼동행렬을 이미지로 저장")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    args = parse_args()
    setup_logging()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델 파일 경로 결정: 인자 > results/best_model.pkl > best_model.pkl
    model_path = Path(args.model) if args.model else Path("results") / "best_model.pkl"
    if not model_path.exists():
        model_path = Path("best_model.pkl")

    logging.info(f"모델 불러오기: {model_path}")
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    # 데이터 불러오기 및 분할
    dataset = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=args.seed,
        stratify=dataset.target,
    )

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"데모 정확도: {accuracy:.4f}")
    logging.info(f"예측 앞 5개: {predictions[:5]}")

    # 혼동행렬 생성 및 저장 옵션
    cm = confusion_matrix(y_test, predictions)
    if args.save_plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.target_names)
        fig, ax = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title(f"Confusion Matrix (acc={accuracy:.4f})")
        plot_path = out_dir / "confusion_matrix.png"
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        logging.info(f"혼동행렬 이미지 저장: {plot_path}")


if __name__ == "__main__":
    main()
