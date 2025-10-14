# modeling/scripts/export_model.py
import mlflow
import os
import shutil

# --- 설정 ---
# MLflow UI에서 복사한 최적 모델의 Run ID
BEST_RUN_ID = "여기에_MLflow_UI에서_찾은_최적_Run_ID를_붙여넣으세요"
# 최종 모델이 저장될 경로 및 이름
EXPORT_MODEL_DIR = "../../app/models/"
EXPORT_MODEL_NAME = "drum_cnn_final.h5"

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # MLflow에서 모델 아티팩트의 로컬 경로를 가져옵니다.
    logged_model_uri = f"runs:/{BEST_RUN_ID}/model/data/model"
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=logged_model_uri)

    # 최종 서빙 폴더로 모델 복사
    os.makedirs(EXPORT_MODEL_DIR, exist_ok=True)
    destination_path = os.path.join(EXPORT_MODEL_DIR, EXPORT_MODEL_NAME)

    shutil.copy(local_model_path, destination_path)

    print(f"모델 추출 완료! Run ID '{BEST_RUN_ID}'의 모델이 아래 경로에 저장되었습니다:")
    print(os.path.abspath(destination_path))