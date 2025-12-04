# core/model_trainer.py

import pandas as pd
import math
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config


class ModelTrainer:
    def __init__(self, all_results_df, elo_results_df):
        self.all_results = all_results_df.sort_values('date').copy()
        self.elo_results = elo_results_df.sort_values('date').copy()

        self.team_elos = {team: config.INITIAL_ELO for team in
                          set(self.elo_results['home_team']).union(set(self.elo_results['away_team']))}

        print("ModelTrainer: ELO puanları hesaplanıyor...")
        self.elo_results.apply(self._process_match_result, axis=1)

    def _calculate_expected_score(self, rating_a, rating_b):
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def _update_elo(self, rating_a, rating_b, actual_score_a, weight_factor):
        expected_score_a = self._calculate_expected_score(rating_a, rating_b)
        k_adjusted = config.K_FACTOR * weight_factor
        return rating_a + k_adjusted * (actual_score_a - expected_score_a)

    def _process_match_result(self, row):
        home_team, away_team = row['home_team'], row['away_team']
        home_score, away_score = row['home_score'], row['away_score']
        league_code = row.get('league_code', 'E0')
        weight = config.LEAGUE_WEIGHTS.get(league_code, 0.5)

        home_elo = self.team_elos.get(home_team, config.INITIAL_ELO)
        away_elo = self.team_elos.get(away_team, config.INITIAL_ELO)

        score_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0

        new_home_elo = self._update_elo(home_elo, away_elo, score_home, weight)
        new_away_elo = self._update_elo(away_elo, home_elo, 1.0 - score_home, weight)

        self.team_elos[home_team] = new_home_elo
        self.team_elos[away_team] = new_away_elo

    def _team_stats_last_n(self, team, before_date, n=5, venue='all'):
        """
        GELİŞMİŞ İSTATİSTİK: Skor, Puan, İY, Şut, Korner
        venue: 'all', 'home', 'away'
        """
        base_cond = (self.all_results['date'] < before_date)

        if venue == 'home':
            cond = base_cond & (self.all_results['home_team'] == team)
        elif venue == 'away':
            cond = base_cond & (self.all_results['away_team'] == team)
        else:
            cond = base_cond & ((self.all_results['home_team'] == team) | (self.all_results['away_team'] == team))

        sub = self.all_results[cond].tail(n)
        current_elo = self.team_elos.get(team, config.INITIAL_ELO)

        if sub.empty:
            return {
                'avg_scored': 0, 'avg_conceded': 0, 'avg_points': 0,
                'ht_avg_scored': 0, 'ht_avg_conceded': 0,
                'avg_shots_target': 0, 'avg_shots_conceded': 0,
                'avg_corners': 0,
                'current_elo': current_elo
            }

        scored, conceded, points = 0, 0, 0
        ht_scored, ht_conceded = 0, 0

        # YENİ METRİKLER
        shots_target_for, shots_target_against = 0, 0
        corners_for = 0

        for _, r in sub.iterrows():
            is_home_for_team = (r['home_team'] == team)

            # Skorlar
            s = r['home_score'] if is_home_for_team else r['away_score']
            c = r['away_score'] if is_home_for_team else r['home_score']
            scored += s;
            conceded += c
            if s > c:
                points += 3
            elif s == c:
                points += 1

            # İlk Yarı
            if r.get('ht_home_score', -1) != -1:
                hts = r['ht_home_score'] if is_home_for_team else r['ht_away_score']
                htc = r['ht_away_score'] if is_home_for_team else r['ht_home_score']
                ht_scored += hts;
                ht_conceded += htc

            # Şut ve Korner (Sıfırdan büyükse topla)
            if r.get('home_shots_target', 0) > 0 or r.get('away_shots_target', 0) > 0:
                st_f = r['home_shots_target'] if is_home_for_team else r['away_shots_target']
                st_a = r['away_shots_target'] if is_home_for_team else r['home_shots_target']
                co_f = r['home_corners'] if is_home_for_team else r['away_corners']

                shots_target_for += st_f
                shots_target_against += st_a
                corners_for += co_f

        count = len(sub)
        return {
            'avg_scored': scored / count,
            'avg_conceded': conceded / count,
            'avg_points': points / count,
            'ht_avg_scored': ht_scored / count,
            'ht_avg_conceded': ht_conceded / count,

            # YENİ: Ortalama İsabetli Şut ve Korner
            'avg_shots_target': shots_target_for / count,
            'avg_shots_conceded': shots_target_against / count,
            'avg_corners': corners_for / count,

            'current_elo': current_elo
        }

    def build_features_for_all_matches(self, last_n=5):
        print("ModelTrainer: Özellikler (Şut/Korner dahil) oluşturuluyor...")
        rows = []
        y_targets = {k: [] for k in ['over15', 'over25', 'over35', 'kg', 'result', 'ht_result', 'ht_over05']}

        for index, r in self.elo_results.iterrows():
            if r.get('ht_home_score', -1) == -1: continue

            home, away = r['home_team'], r['away_team']
            date = r['date']

            # 1. GENEL FORM
            h_gen = self._team_stats_last_n(home, date, n=5, venue='all')
            a_gen = self._team_stats_last_n(away, date, n=5, venue='all')

            # 2. İÇ/DIŞ SAHA FORMU
            h_home = self._team_stats_last_n(home, date, n=5, venue='home')
            a_away = self._team_stats_last_n(away, date, n=5, venue='away')

            features = {
                # ELO
                'elo_diff': h_gen['current_elo'] - a_gen['current_elo'],
                'elo_home': h_gen['current_elo'],
                'elo_away': a_gen['current_elo'],

                # Puan ve Gol (Genel)
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

                # İY İstatistikleri
                'h_ht_scored': h_home['ht_avg_scored'],
                'a_ht_scored': a_away['ht_avg_scored'],
                'h_ht_conceded': h_home['ht_avg_conceded'],
                'a_ht_conceded': a_away['ht_avg_conceded']
            }
            rows.append(features)

            # Hedefler
            total = r['home_score'] + r['away_score']
            y_targets['over15'].append(1 if total > 1.5 else 0)
            y_targets['over25'].append(1 if total > 2.5 else 0)
            y_targets['over35'].append(1 if total > 3.5 else 0)
            y_targets['kg'].append(1 if r['home_score'] > 0 and r['away_score'] > 0 else 0)

            if r['home_score'] > r['away_score']:
                y_targets['result'].append(2)
            elif r['home_score'] == r['away_score']:
                y_targets['result'].append(1)
            else:
                y_targets['result'].append(0)

            ht_total = r['ht_home_score'] + r['ht_away_score']
            y_targets['ht_over05'].append(1 if ht_total > 0.5 else 0)

            if r['ht_home_score'] > r['ht_away_score']:
                y_targets['ht_result'].append(2)
            elif r['ht_home_score'] == r['ht_away_score']:
                y_targets['ht_result'].append(1)
            else:
                y_targets['ht_result'].append(0)

        X = pd.DataFrame(rows)
        return X, y_targets

    def train_and_save_all(self, X, y_dict):
        print("ModelTrainer: Modeller eğitiliyor (Şut ve Korner verileriyle)...")
        os.makedirs(config.MODELS_FOLDER, exist_ok=True)

        for name, y in y_dict.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
            model.fit(X_train, y_train)

            acc = accuracy_score(y_test, model.predict(X_test))
            print(f"  -> {name.upper()} Modeli Başarısı: %{acc * 100:.2f}")
            joblib.dump(model, os.path.join(config.MODELS_FOLDER, f"{name}.joblib"))

    def get_last_n_matches(self, team, n=5):
        cond = (self.all_results['home_team'] == team) | (self.all_results['away_team'] == team)
        return self.all_results[cond].tail(n)