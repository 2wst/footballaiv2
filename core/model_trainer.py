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
                'avg_shots_target': 0, 'avg_shots_conceded': 0, 'avg_corners': 0,
                'current_elo': current_elo
            }

        scored, conceded, points = 0, 0, 0
        ht_scored, ht_conceded = 0, 0
        shots_target_for, shots_target_against = 0, 0
        corners_for = 0

        for _, r in sub.iterrows():
            is_home_for_team = (r['home_team'] == team)
            s = r['home_score'] if is_home_for_team else r['away_score']
            c = r['away_score'] if is_home_for_team else r['home_score']
            scored += s; conceded += c
            if s > c: points += 3
            elif s == c: points += 1
            
            if r.get('ht_home_score', -1) != -1:
                hts = r['ht_home_score'] if is_home_for_team else r['ht_away_score']
                htc = r['ht_away_score'] if is_home_for_team else r['ht_home_score']
                ht_scored += hts; ht_conceded += htc
            
            if r.get('home_shots_target', 0) > 0 or r.get('away_shots_target', 0) > 0:
                st_f = r['home_shots_target'] if is_home_for_team else r['away_shots_target']
                st_a = r['away_shots_target'] if is_home_for_team else r['home_shots_target']
                co_f = r['home_corners'] if is_home_for_team else r['away_corners']
                shots_target_for += st_f; shots_target_against += st_a; corners_for += co_f

        count = len(sub)
        return {
            'avg_scored': scored / count, 'avg_conceded': conceded / count, 'avg_points': points / count,
            'ht_avg_scored': ht_scored / count, 'ht_avg_conceded': ht_conceded / count,
            'avg_shots_target': shots_target_for / count, 'avg_shots_conceded': shots_target_against / count,
            'avg_corners': corners_for / count, 'current_elo': current_elo
        }

    # --- YENİ FONKSİYON: H2H İSTATİSTİKLERİ ---
    def _get_h2h_stats(self, home_team, away_team, before_date, n=5):
        """İki takım arasındaki son n maçı analiz eder."""
        cond = (self.all_results['date'] < before_date) & \
               (((self.all_results['home_team'] == home_team) & (self.all_results['away_team'] == away_team)) | \
                ((self.all_results['home_team'] == away_team) & (self.all_results['away_team'] == home_team)))
        
        sub = self.all_results[cond].tail(n)
        
        if sub.empty:
            return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0, 'h2h_avg_goals': 2.5} # Varsayılan

        home_wins, away_wins, draws = 0, 0, 0
        total_goals = 0

        for _, r in sub.iterrows():
            total_goals += (r['home_score'] + r['away_score'])
            if r['home_score'] == r['away_score']:
                draws += 1
            elif r['home_team'] == home_team: # Ev sahibi bizim Home ise
                if r['home_score'] > r['away_score']: home_wins += 1
                else: away_wins += 1
            else: # Ev sahibi bizim Away ise (yani maç ters oynanmışsa)
                if r['home_score'] > r['away_score']: away_wins += 1
                else: home_wins += 1
        
        count = len(sub)
        return {
            'h2h_home_wins': home_wins / count, # Oran olarak döndür
            'h2h_away_wins': away_wins / count,
            'h2h_draws': draws / count,
            'h2h_avg_goals': total_goals / count
        }

    def build_features_for_all_matches(self, last_n=5):
        print("ModelTrainer: Özellikler (H2H + Şut + ELO) oluşturuluyor...")
        rows = []
        y_targets = {k: [] for k in ['over15', 'over25', 'over35', 'kg', 'result', 'ht_result', 'ht_over05']}

        for index, r in self.elo_results.iterrows():
            if r.get('ht_home_score', -1) == -1: continue 

            home, away = r['home_team'], r['away_team']
            date = r['date']

            # Temel Stats
            h_gen = self._team_stats_last_n(home, date, n=5, venue='all')
            a_gen = self._team_stats_last_n(away, date, n=5, venue='all')
            h_home = self._team_stats_last_n(home, date, n=5, venue='home')
            a_away = self._team_stats_last_n(away, date, n=5, venue='away')

            # H2H Stats (YENİ)
            h2h = self._get_h2h_stats(home, away, date, n=5)

            features = {
                # ELO
                'elo_diff': h_gen['current_elo'] - a_gen['current_elo'],
                'elo_home': h_gen['current_elo'], 'elo_away': a_gen['current_elo'],
                
                # Genel & Yerel
                'h_form': h_gen['avg_points'], 'a_form': a_gen['avg_points'],
                'h_home_pts': h_home['avg_points'], 'a_away_pts': a_away['avg_points'],
                
                'h_att_str': (h_gen['avg_scored'] + h_home['avg_scored']) / 2, # Hücum Gücü
                'a_def_weak': (a_gen['avg_conceded'] + a_away['avg_conceded']) / 2, # Defans Zaafı
                
                'h_sot': h_home['avg_shots_target'], 'a_sot': a_away['avg_shots_target'],
                'h_corn': h_home['avg_corners'], 'a_corn': a_away['avg_corners'],

                # H2H (Aralarındaki Maçlar) - KRİTİK
                'h2h_home_win_rate': h2h['h2h_home_wins'],
                'h2h_away_win_rate': h2h['h2h_away_wins'],
                'h2h_avg_goals': h2h['h2h_avg_goals']
            }
            rows.append(features)

            # Hedefler
            total = r['home_score'] + r['away_score']
            y_targets['over15'].append(1 if total > 1.5 else 0)
            y_targets['over25'].append(1 if total > 2.5 else 0)
            y_targets['over35'].append(1 if total > 3.5 else 0)
            y_targets['kg'].append(1 if r['home_score'] > 0 and r['away_score'] > 0 else 0)
            if r['home_score'] > r['away_score']: y_targets['result'].append(2)
            elif r['home_score'] == r['away_score']: y_targets['result'].append(1)
            else: y_targets['result'].append(0)
            
            ht_total = r['ht_home_score'] + r['ht_away_score']
            y_targets['ht_over05'].append(1 if ht_total > 0.5 else 0)
            if r['ht_home_score'] > r['ht_away_score']: y_targets['ht_result'].append(2)
            elif r['ht_home_score'] == r['ht_away_score']: y_targets['ht_result'].append(1)
            else: y_targets['ht_result'].append(0)

        X = pd.DataFrame(rows)
        return X, y_targets

    def train_and_save_all(self, X, y_dict):
        print("ModelTrainer: Modeller optimize ediliyor...")
        os.makedirs(config.MODELS_FOLDER, exist_ok=True)
        
        for name, y in y_dict.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Parametre İyileştirmesi: Daha yavaş öğren, daha çok ağaç kur (Overfitting engeller, genellemeyi artırır)
            model = GradientBoostingClassifier(
                n_estimators=300,       # Daha fazla ağaç
                learning_rate=0.03,     # Daha hassas öğrenme
                max_depth=3,            # Çok derinleşme (Ezberleme)
                subsample=0.8,          # Her ağaçta verinin %80'ini kullan (Varyansı azaltır)
                random_state=42
            )
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            print(f"  -> {name.upper()} Başarısı: %{acc*100:.2f}")
            joblib.dump(model, os.path.join(config.MODELS_FOLDER, f"{name}.joblib"))

    def get_last_n_matches(self, team, n=5):
        cond = (self.all_results['home_team'] == team) | (self.all_results['away_team'] == team)
        return self.all_results[cond].tail(n)
