# predict.py
import argparse
from core.data_manager import DataManager
from core.model_trainer import ModelTrainer
from core.predictor import MatchPredictor

def main():
    parser = argparse.ArgumentParser(description="İki takım için maç tahmini yapar.")
    parser.add_argument("home_team", type=str, help="Ev sahibi takımın adı")
    parser.add_argument("away_team", type=str, help="Deplasman takımının adı")
    args = parser.parse_args()

    print("Backend test ediliyor: Gerekli veriler ve modeller yükleniyor...")

    try:
        # 1. Veri yöneticisini başlat ve veriyi yükle
        data_manager = DataManager()
        all_results_df, elo_results_df = data_manager.load_all_data()

        # 2. ELO ve istatistik hesaplamaları için ModelTrainer'ı yükle
        trainer = ModelTrainer(all_results_df, elo_results_df)

        # 3. Tahmin yapmak için MatchPredictor'ı yükle
        predictor = MatchPredictor(trainer)

        print("\n✅ Veriler ve modeller başarıyla yüklendi. Tahmin yapılıyor...")

        # 4. Tahmini yap ve sonuçları al
        prediction, home_elo, away_elo = predictor.predict_match(args.home_team, args.away_team)

        # ... (geri kalan kısım aynı) ...

    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
        print("Lütfen önce modelleri eğittiğinizden emin olun: python train.py --train")
    except Exception as e:
        print(f"\nBeklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    main()