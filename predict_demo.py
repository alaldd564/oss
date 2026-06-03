from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="저장된 모델로 데모 예측 및 혼동행렬 시각화")
    parser.add_argument("--model", type=str, default=None, help="불러올 모델 파일 경로")
    parser.add_argument("--out", type=str, default=".", help="결과 출력 디렉터리")
    parser.add_argument("--seed", type=int, default=42, help="데이터 분할 시드")
    parser.add_argument("--save-plot", action="store_true", help="혼동행렬을 이미지로 저장")
    parser.add_argument("--normalize-cm", action="store_true", help="혼동행렬을 정규화하여 이미지로 저장")
    parser.add_argument("--font-size", type=int, default=9, help="혼동행렬 텍스트 폰트 크기 (기본: 9)")
    parser.add_argument("--plot-roc", action="store_true", help="다중 클래스 ROC 플롯을 생성")
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
        labels = dataset.target_names
        # 기본 혼동행렬
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title(f"Confusion Matrix (acc={accuracy:.4f})")
        plot_path = out_dir / "confusion_matrix.png"
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        logging.info(f"혼동행렬 이미지 저장: {plot_path}")

        if args.normalize_cm:
            # 정규화된(비율) 혼동행렬
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(6, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
            disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
            plt.title(f"Confusion Matrix (normalized)")
            for t in ax.texts:
                t.set_fontsize(args.font_size)
            norm_path = out_dir / "confusion_matrix_normalized.png"
            plt.tight_layout()
            fig.savefig(norm_path)
            plt.close(fig)
            logging.info(f"정규화 혼동행렬 저장: {norm_path}")

    # ROC 플롯 (다중 클래스) 옵션
    if args.plot_roc:
        try:
            # 다중 클래스를 위한 바이너리화
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc

            classes = np.arange(len(dataset.target_names))
            y_test_b = label_binarize(y_test, classes=classes)
            # 예측 확률 얻기
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(x_test)
            else:
                # 일부 모델/파이프라인은 단계에 따라 다를 수 있음
                y_score = np.vstack([np.zeros(len(x_test)) + (model.predict(x_test) == i) for i in classes]).T

            # 각 클래스별 ROC 및 AUC
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, name in enumerate(dataset.target_names):
                fpr, tpr, _ = roc_curve(y_test_b[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")

            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Multi-class ROC")
            ax.legend(loc="lower right")
            roc_path = out_dir / "roc_curves.png"
            plt.tight_layout()
            fig.savefig(roc_path)
            plt.close(fig)
            logging.info(f"ROC 플롯 저장: {roc_path}")
        except Exception as e:
            logging.warning(f"ROC 플롯 생성 실패: {e}")


if __name__ == "__main__":
    main()
