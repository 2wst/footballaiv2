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
                raise FileNotFoundError(f"Model bulunamadı! Lütfen 'python train.py --train' çalıştırın.")
            models[name] = joblib.load(path)
        return models

    def predict_match(self, home_team: str, away_team: str, last_n=5):
        today = pd.to_datetime('today')

        # Stats
        h_gen = self.trainer._team_stats_last_n(home_team, today, n=5, venue='all')
        a_gen = self.trainer._team_stats_last_n(away_team, today, n=5, venue='all')
        h_home = self.trainer._team_stats_last_n(home_team, today, n=5, venue='home')
        a_away = self.trainer._team_stats_last_n(away_team, today, n=5, venue='away')
        
        # H2H (YENİ)
        h2h = self.trainer._get_h2h_stats(home_team, away_team, today, n=5)

        home_elo = h_gen['current_elo']
        away_elo = a_gen['current_elo']

        # ÖZELLİKLER (Train ile AYNI İSİM VE SIRA)
        features = pd.DataFrame([{
            'elo_diff': home_elo - away_elo,
            'elo_home': home_elo, 'elo_away': away_elo,
            
            'h_form': h_gen['avg_points'], 'a_form': a_gen['avg_points'],
            'h_home_pts': h_home['avg_points'], 'a_away_pts': a_away['avg_points'],
            
            'h_att_str': (h_gen['avg_scored'] + h_home['avg_scored']) / 2,
            'a_def_weak': (a_gen['avg_conceded'] + a_away['avg_conceded']) / 2,
            
            'h_sot': h_home['avg_shots_target'], 'a_sot': a_away['avg_shots_target'],
            'h_corn': h_home['avg_corners'], 'a_corn': a_away['avg_corners'],

            'h2h_home_win_rate': h2h['h2h_home_wins'],
            'h2h_away_win_rate': h2h['h2h_away_wins'],
            'h2h_avg_goals': h2h['h2h_avg_goals']
        }])

        # Tahminler (Calibration ile)
        raw_over15 = float(self.models['over15'].predict_proba(features)[0, 1])
        raw_over25 = float(self.models['over25'].predict_proba(features)[0, 1])
        raw_over35 = float(self.models['over35'].predict_proba(features)[0, 1])

        # Mantıksal Düzeltme
        if raw_over35 > raw_over25: raw_over35 = raw_over25 - 0.02
        if raw_over25 > raw_over15: raw_over25 = raw_over15 - 0.02
        if raw_over35 > raw_over25: raw_over35 = raw_over25 - 0.02

        out = {}
        out['over15'] = raw_over15; out['under15'] = 1.0 - raw_over15
        out['over25'] = raw_over25; out['under25'] = 1.0 - raw_over25
        out['over35'] = raw_over35; out['under35'] = 1.0 - raw_over35
        
        out['kg_var'] = float(self.models['kg'].predict_proba(features)[0, 1])
        out['kg_yok'] = 1.0 - out['kg_var']
        
        probs_res = self.models['result'].predict_proba(features)[0]
        out['home_win'] = probs_res[2]; out['draw'] = probs_res[1]; out['away_win'] = probs_res[0]
        
        probs_ht = self.models['ht_result'].predict_proba(features)[0]
        out['ht_home'] = probs_ht[2]; out['ht_draw'] = probs_ht[1]; out['ht_away'] = probs_ht[0]
        out['ht_over05'] = float(self.models['ht_over05'].predict_proba(features)[0, 1])
        out['ht_under05'] = 1.0 - out['ht_over05']

        return out, home_elo, away_elo
