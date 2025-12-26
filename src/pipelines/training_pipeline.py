import os
import pickle
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import mlflow
import mlflow.sklearn
from src.utils.common import read_params
from src.utils.logger import get_logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.components.data_ingestion import load_data
from src.components.data_transformation import ColumnDropper, MissingValueImputer, CategoricalEncoder

logger = get_logger(__name__)

# --- MLFLOW AYARLARI ---
# Docker içinden "http://mlflow:5000", lokalden "http://localhost:5000" okur.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Titanic_Experiment")


def train_model(config_path):
    config = read_params(config_path)

    # Base directory: Proje ana dizinini bulur (src/pipelines/training_pipeline.py -> root)
    # Bu sayede kodu nereden çalıştırırsan çalıştır yollar bozulmaz.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # YAML'dan yolları oku ve ana dizinle birleştir
    data_path = os.path.join(base_dir, config['external_data_config']['external_data_csv'])
    model_dir = os.path.join(base_dir, config['model_config']['model_dir'])
    model_name = config['model_config']['model_name']
    model_path = os.path.join(model_dir, model_name)

    # Parametreleri YAML'dan Oku
    random_state = config['preprocessing_config']['random_state']
    split_ratio = config['preprocessing_config']['train_test_split_ratio']

    n_estimators = config['model_config']['n_estimators']
    max_depth = config['model_config']['max_depth']
    # DÜZELTME: YAML'da adı 'random_state', kodda da öyle çektik
    model_random_state = config['model_config']['random_state']

    logger.info(f"Veri yükleniyor: {data_path}")

    if not os.path.exists(data_path):
        logger.error(f"HATA: Veri dosyası bulunamadı -> {data_path}")
        raise FileNotFoundError(f"{data_path} bulunamadı. Lütfen 'data/raw' klasörünü kontrol et.")

    df = load_data(data_path)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_ratio,
        random_state=random_state,
        stratify=y
    )

    # --- MLFLOW RUN BAŞLATIYORUZ ---
    with mlflow.start_run():
        logger.info("MLflow run başlatıldı... 🧪")

        pipeline = Pipeline([
            ('dropper', ColumnDropper(columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin'])),
            ('imputer', MissingValueImputer()),
            ('encoder', CategoricalEncoder()),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=model_random_state
            ))
        ])

        # MLflow Parametre Kaydı
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("split_ratio", split_ratio)
        mlflow.log_param("model_type", "RandomForestClassifier")

        logger.info(f"Model eğitiliyor... (n_estimators={n_estimators}, max_depth={max_depth})")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy Değeri: {accuracy}")

        # MLflow Metrik Kaydı
        mlflow.log_metric("accuracy", accuracy)

        # MLflow Model Registry (Modeli buluta/sunucuya kaydet)
        mlflow.sklearn.log_model(pipeline, "model")
        logger.info("Model MLflow veritabanına kaydedildi. 🚀")

        # --- LOKAL YEDEKLEME (API Kullanımı İçin) ---
        logger.info(f"Model lokal diske yedekleniyor: {model_path}")
        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)

        logger.info(f"Pipeline başarıyla tamamlandı! ✅")


if __name__ == "__main__":
    # params.yaml dosyasını dinamik olarak bul
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 3 seviye yukarı çık: src/pipelines -> src -> root
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))

    config_path = os.path.join(project_root, "params.yaml")

    print(f"Config dosyası aranıyor: {config_path}")

    if not os.path.exists(config_path):
        print("UYARI: params.yaml tam yolda bulunamadı, varsayılan 'params.yaml' deneniyor.")
        config_path = "params.yaml"

    train_model(config_path)