# core/data_manager.py

import pandas as pd
import os
import time
import requests
import config
from core.utils import clean_team_name

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}


class DataManager:
    def __init__(self, league_codes=config.LEAGUE_CODES):
        self.league_codes = league_codes

    def _scrape_footballdata_data(self, league_code: str):
        """
        football-data.co.uk'ten detaylı istatistiklerle veri çeker.
        """
        all_season_dfs = []
        current_year_short = int(time.strftime("%y"))
        seasons_to_check = [
            f"{year - 1}{year}" for year in range(current_year_short + 1, current_year_short - 3, -1)
        ]

        print(f" football-data.co.uk'ten '{league_code}' verileri çekiliyor...")
        for season in seasons_to_check:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
                df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')

                # YENİ: Şut (HS/AS), İsabetli Şut (HST/AST) ve Korner (HC/AC) eklendi
                rename_map = {
                    'Date': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
                    'FTHG': 'home_score', 'FTAG': 'away_score',
                    'HTHG': 'ht_home_score', 'HTAG': 'ht_away_score',
                    'HS': 'home_shots', 'AS': 'away_shots',
                    'HST': 'home_shots_target', 'AST': 'away_shots_target',
                    'HC': 'home_corners', 'AC': 'away_corners'
                }

                # Sütunları kontrol et ve seç
                existing_cols = list(set(rename_map.keys()) & set(df.columns))
                if not {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}.issubset(df.columns):
                    continue

                df = df[existing_cols].copy()
                df = df.rename(columns=rename_map)

                df['league_code'] = league_code
                df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'], inplace=True)
                all_season_dfs.append(df)

                print(f"  -> {league_code} - {season} sezonu başarıyla çekildi.")
                time.sleep(1)

            except Exception as e:
                continue

        if not all_season_dfs: return pd.DataFrame()
        return pd.concat(all_season_dfs, ignore_index=True)

    def _scrape_international_data(self):
        """
        Milli maç verisi (Detaylı istatistik içermez, sadece skor).
        """
        code = 'INT'
        url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

        try:
            print(f"  -> {code} (Milli Maç) verisi GitHub'dan çekiliyor...")
            df = pd.read_csv(url, on_bad_lines='skip')

            required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
            if not set(required_cols).issubset(df.columns):
                return pd.DataFrame()

            df = df[required_cols].copy()
            df['league_code'] = code

            df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
            df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
            df.dropna(subset=['home_score', 'away_score'], inplace=True)

            # Milli maçlarda detay verisi olmadığı için NaN/Exksi değer kalacak, load_all_data'da 0'layacağız
            return df
        except Exception as e:
            print(f"  -> HATA: {code} verisi çekilemedi: {e}")
            return pd.DataFrame()

    def load_all_data(self):
        cleaned_dataframes = []
        print("📁 Veri kaynakları işleniyor...")
        for code in self.league_codes:
            df = pd.DataFrame()
            if code == 'INT':
                df = self._scrape_international_data()
            else:
                df = self._scrape_footballdata_data(code)
            if not df.empty:
                cleaned_dataframes.append(df)

        if not cleaned_dataframes: raise ValueError("Hiçbir veri kaynağı başarıyla işlenemedi.")

        final_df = pd.concat(cleaned_dataframes, ignore_index=True)

        print("🧹 Son ortak temizlik ve formatlama yapılıyor...")
        final_df['home_team'] = final_df['home_team'].apply(clean_team_name)
        final_df['away_team'] = final_df['away_team'].apply(clean_team_name)

        # Tarih formatlama
        mask_int = final_df['league_code'] == 'INT'
        final_df.loc[~mask_int, 'date'] = pd.to_datetime(final_df.loc[~mask_int, 'date'], errors='coerce',
                                                         dayfirst=True)
        final_df.loc[mask_int, 'date'] = pd.to_datetime(final_df.loc[mask_int, 'date'], errors='coerce')
        final_df['date'] = pd.to_datetime(final_df['date'])

        # Eksik İY skorlarını -1 yap
        if 'ht_home_score' not in final_df.columns:
            final_df['ht_home_score'] = -1
            final_df['ht_away_score'] = -1
        else:
            final_df['ht_home_score'] = final_df['ht_home_score'].fillna(-1).astype(int)
            final_df['ht_away_score'] = final_df['ht_away_score'].fillna(-1).astype(int)

        # YENİ: Detaylı istatistik sütunlarını doldur (Yoksa 0 yap)
        stat_cols = ['home_shots', 'away_shots', 'home_shots_target', 'away_shots_target', 'home_corners',
                     'away_corners']
        for col in stat_cols:
            if col not in final_df.columns:
                final_df[col] = 0
            else:
                final_df[col] = final_df[col].fillna(0).astype(int)

        final_df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'], inplace=True)
        final_df['home_score'] = final_df['home_score'].astype(int)
        final_df['away_score'] = final_df['away_score'].astype(int)

        # Tarih Filtresi (Backtest)
        if config.CUTOFF_DATE:
            print(f"✂️ Backtest Modu: Veriler {config.CUTOFF_DATE} tarihine kadar alınıyor.")
            final_df = final_df[final_df['date'] <= config.CUTOFF_DATE]

        # ELO Filtresi
        print("🔍 Elo hesaplaması için maçlar filtreleniyor...")
        national_matches = final_df[(final_df['league_code'] == 'INT') & (final_df['date'].dt.year >= 2015)]
        club_matches = final_df[(final_df['league_code'] != 'INT') & (final_df['date'] >= '2019-08-01')]

        elo_filtered_df = pd.concat([national_matches, club_matches], ignore_index=True).sort_values(
            by='date').reset_index(drop=True)

        print(f"✨ ELO hesaplaması için {len(elo_filtered_df)} maç kullanılacak.")
        return final_df, elo_filtered_df