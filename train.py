# train.py
import argparse
from core.data_manager import DataManager
from core.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Futbol tahmin modellerini eğitir.")
    parser.add_argument("--train", action="store_true", help="Verilerden modelleri eğitir.")
    args = parser.parse_args()

    if args.train:
        # 1. Veriyi yükle (artık 2 dataframe dönüyor)
        data_manager = DataManager()
        all_results_df, elo_results_df = data_manager.load_all_data()

        # 2. Model eğiticiyi iki dataframe ile başlat
        trainer = ModelTrainer(all_results_df, elo_results_df)

        # 3. Özellikleri oluştur
        X, y_dict = trainer.build_features_for_all_matches()

        # 4. Modelleri eğit ve kaydet
        trainer.train_and_save_all(X, y_dict)
    else:
        print("Modeli eğitmek için '--train' argümanını kullanın.")
        print("Örnek: python train.py --train")

if __name__ == "__main__":
    main()
