import argparse

from download_artifacts import download_s3_folder
from export_classifier_to_onnx import export_classifier_to_onnx
from export_sentence_transformer_to_onnx import export_model_to_onnx
from sentiment_app.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-artifacts", action="store_true", help="Download artifacts from S3")
    parser.add_argument("--export-models", action="store_true", help="Export models to ONNX")

    args = parser.parse_args()
    settings = Settings()

    if args.download_artifacts:
        download_s3_folder(settings)

    if args.export_models:
        export_classifier_to_onnx(settings)
        export_model_to_onnx(settings)
