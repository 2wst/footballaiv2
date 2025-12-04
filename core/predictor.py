# core/predictor.py

import pandas as pd
import joblib
import os
import config
from core.model_trainer import ModelTrainer


class MatchPredictor:
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        model_names = ['over15', 'over25', 'over35', 'kg', 'result', 'ht_result', 'ht_over05']
        for name in model_names:
            path = os.path.join(config.MODELS_FOLDER, f"{name}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model {path} bulunamadı! Lütfen 'python train.py --train' çalıştırın.")
            models[name] = joblib.load(path)
        return models

    def predict_match(self, home_team: str, away_team: str, last_n=5):
        today = pd.to_datetime('today')

        # İstatistikleri hesapla
        h_gen = self.trainer._team_stats_last_n(home_team, today, n=5, venue='all')
        a_gen = self.trainer._team_stats_last_n(away_team, today, n=5, venue='all')

        h_home = self.trainer._team_stats_last_n(home_team, today, n=5, venue='home')
        a_away = self.trainer._team_stats_last_n(away_team, today, n=5, venue='away')

        home_elo = h_gen['current_elo']
        away_elo = a_gen['current_elo']

        # ÖZELLİK TABLOSU (Sırası train_and_save_all ile birebir aynı)
        features = pd.DataFrame([{
            # ELO
            'elo_diff': home_elo - away_elo,
            'elo_home': home_elo,
            'elo_away': away_elo,

            # Genel
            'h_gen_points': h_gen['avg_points'],
            'a_gen_points': a_gen['avg_points'],
            'h_gen_scored': h_gen['avg_scored'],
            'a_gen_scored': a_gen['avg_scored'],

            # Ev Sahibi (İç Saha)
            'h_home_points': h_home['avg_points'],
            'h_home_scored': h_home['avg_scored'],
            'h_home_conceded': h_home['avg_conceded'],
            'h_home_sot': h_home['avg_shots_target'],  # YENİ
            'h_home_sot_c': h_home['avg_shots_conceded'],  # YENİ
            'h_home_corn': h_home['avg_corners'],  # YENİ

            # Deplasman (Dış Saha)
            'a_away_points': a_away['avg_points'],
            'a_away_scored': a_away['avg_scored'],
            'a_away_conceded': a_away['avg_conceded'],
            'a_away_sot': a_away['avg_shots_target'],  # YENİ
            'a_away_sot_c': a_away['avg_shots_conceded'],  # YENİ
            'a_away_corn': a_away['avg_corners'],  # YENİ

            # İY
            'h_ht_scored': h_home['ht_avg_scored'],
            'a_ht_scored': a_away['ht_avg_scored'],
            'h_ht_conceded': h_home['ht_avg_conceded'],
            'a_ht_conceded': a_away['ht_avg_conceded']
        }])

        # --- TAHMİNLER ---
        out = {}

        # Goller
        out['over15'] = float(self.models['over15'].predict_proba(features)[0, 1])
        out['under15'] = 1.0 - out['over15']
        out['over25'] = float(self.models['over25'].predict_proba(features)[0, 1])
        out['under25'] = 1.0 - out['over25']
        out['over35'] = float(self.models['over35'].predict_proba(features)[0, 1])
        out['under35'] = 1.0 - out['over35']

        # KG
        out['kg_var'] = float(self.models['kg'].predict_proba(features)[0, 1])
        out['kg_yok'] = 1.0 - out['kg_var']

        # Maç Sonucu
        probs_res = self.models['result'].predict_proba(features)[0]
        out['home_win'] = probs_res[2]
        out['draw'] = probs_res[1]
        out['away_win'] = probs_res[0]

        # İY Sonucu
        probs_ht = self.models['ht_result'].predict_proba(features)[0]
        out['ht_home'] = probs_ht[2]
        out['ht_draw'] = probs_ht[1]
        out['ht_away'] = probs_ht[0]

        out['ht_over05'] = float(self.models['ht_over05'].predict_proba(features)[0, 1])
        out['ht_under05'] = 1.0 - out['ht_over05']

        return out, home_elo, away_elo