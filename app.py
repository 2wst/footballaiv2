import streamlit as st
import pandas as pd
import os
import config

# Gerekli importlar
from core.data_manager import DataManager
from core.model_trainer import ModelTrainer
from core.predictor import MatchPredictor
from api_client import get_todays_matches_by_league

# ===================== #
#   SAYFA AYARLARI      #
# ===================== #
st.set_page_config(page_title="⚽ Futbol AI Tahmin", layout="wide")
st.title("⚽ Futbol Yapay Zeka Tahmincisi")


# ===================== #
#   YÜKLEME             #
# ===================== #
@st.cache_resource(ttl=3600)
def load_dependencies():
    if not os.path.exists(config.MODELS_FOLDER):
        return None, None
    data_manager = DataManager()
    all_results_df, elo_results_df = data_manager.load_all_data()
    trainer = ModelTrainer(all_results_df, elo_results_df)
    predictor = MatchPredictor(trainer)
    return predictor, all_results_df


@st.cache_data(ttl=600)
def fetch_daily_matches(league_code):
    return get_todays_matches_by_league(league_code)


with st.spinner('Modeller yükleniyor...'):
    predictor, results = load_dependencies()

if not predictor:
    st.error("Modeller bulunamadı! Önce 'python train.py --train' çalıştırın.")
    st.stop()

# ===================== #
#   MENÜ VE SEÇİMLER    #
# ===================== #
league_names = {
    'CL': '🏆 Şampiyonlar Ligi', 'B1': '🇧🇪 Belçika', 'D1': '🇩🇪 Bundesliga',
    'E0': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier Lig', 'F1': '🇫🇷 Ligue 1', 'G1': '🇬🇷 Yunanistan',
    'I1': '🇮🇹 Serie A', 'INT': '🌐 Milli Maçlar', 'N1': '🇳🇱 Eredivisie',
    'P1': '🇵🇹 Portekiz', 'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya', 'SP1': '🇪🇸 La Liga', 'T1': '🇹🇷 Süper Lig'
}

st.sidebar.header("Ayarlar")
available_codes = sorted(list(set(results['league_code'].unique()) & set(config.LEAGUE_CODES)))
available_leagues = [league_names.get(code, code) for code in available_codes]

selected_league_name = st.sidebar.selectbox("Lig Seç:", available_leagues)
selected_code = next((code for code, name in league_names.items() if name == selected_league_name), None)

if selected_code:
    league_matches = results[results['league_code'] == selected_code]
    teams = sorted(list(set(league_matches["home_team"].unique()) | set(league_matches["away_team"].unique())))

    st.sidebar.markdown("---")
    home_team = st.sidebar.selectbox("Ev Sahibi", teams)
    away_team = st.sidebar.selectbox("Deplasman", teams, index=min(1, len(teams) - 1))

    predict_btn = st.sidebar.button("ANALİZ ET", type="primary", use_container_width=True)

# ===================== #
#   ANALİZ EKRANI       #
# ===================== #

if selected_code and predict_btn:
    if home_team == away_team:
        st.error("Aynı takımı seçemezsiniz.")
    else:
        with st.spinner("Tüm modeller çalıştırılıyor..."):
            out, home_elo, away_elo = predictor.predict_match(home_team, away_team)

        # 1. BAŞLIK
        st.markdown(f"### {home_team} 🆚 {away_team}")


        # 2. TÜM MODELLER TABLOSU
        # İstediğiniz tüm keys'leri burada satır satır işliyoruz

        table_data = [
            # 1. RESULT (Maç Sonucu)
            {
                "Kategori": "MAÇ SONUCU (MS)",
                "Tahmin 1": f"EV: %{out['home_win'] * 100:.1f}",
                "Tahmin 2": f"BER: %{out['draw'] * 100:.1f}",
                "Tahmin 3": f"DEP: %{out['away_win'] * 100:.1f}"
            },
            # 2. HT_RESULT (İlk Yarı Sonucu)
            {
                "Kategori": "İLK YARI SONUCU (İY)",
                "Tahmin 1": f"İY 1: %{out['ht_home'] * 100:.1f}",
                "Tahmin 2": f"İY 0: %{out['ht_draw'] * 100:.1f}",
                "Tahmin 3": f"İY 2: %{out['ht_away'] * 100:.1f}"
            },
            # 3. KG (Karşılıklı Gol)
            {
                "Kategori": "KARŞILIKLI GOL",
                "Tahmin 1": f"VAR: %{out['kg_var'] * 100:.1f}",
                "Tahmin 2": f"YOK: %{out['kg_yok'] * 100:.1f}",
                "Tahmin 3": "-"
            },
            # 4. HT_OVER05 (İlk Yarı 0.5 Üst)
            {
                "Kategori": "İLK YARI 0.5 GOL",
                "Tahmin 1": f"ÜST: %{out['ht_over05'] * 100:.1f}",
                "Tahmin 2": f"ALT: %{out['ht_under05'] * 100:.1f}",
                "Tahmin 3": "-"
            },
            # 5. OVER15 (1.5 Üst)
            {
                "Kategori": "1.5 GOL BARAJI",
                "Tahmin 1": f"ÜST: %{out['over15'] * 100:.1f}",
                "Tahmin 2": f"ALT: %{out['under15'] * 100:.1f}",
                "Tahmin 3": "-"
            },
            # 6. OVER25 (2.5 Üst)
            {
                "Kategori": "2.5 GOL BARAJI",
                "Tahmin 1": f"ÜST: %{out['over25'] * 100:.1f}",
                "Tahmin 2": f"ALT: %{out['under25'] * 100:.1f}",
                "Tahmin 3": "-"
            },
            # 7. OVER35 (3.5 Üst)
            {
                "Kategori": "3.5 GOL BARAJI",
                "Tahmin 1": f"ÜST: %{out['over35'] * 100:.1f}",
                "Tahmin 2": f"ALT: %{out['under35'] * 100:.1f}",
                "Tahmin 3": "-"
            }
        ]

        # Tabloyu oluştur
        st.table(pd.DataFrame(table_data).set_index("Kategori"))

        # 3. SON MAÇLAR
        st.markdown("---")
        c1, c2 = st.columns(2)
        col_rename = {'date': 'Tarih', 'home_team': 'Ev', 'away_team': 'Dep', 'home_score': 'S1', 'away_score': 'S2'}

        with c1:
            st.markdown(f"**{home_team} Son 5 Maç**")
            last5_h = predictor.trainer.get_last_n_matches(home_team)
            if not last5_h.empty:
                df_h = last5_h.copy()
                df_h['date'] = pd.to_datetime(df_h['date']).dt.strftime('%d/%m')
                st.dataframe(
                    df_h[['date', 'home_team', 'away_team', 'home_score', 'away_score']].rename(columns=col_rename),
                    hide_index=True)

        with c2:
            st.markdown(f"**{away_team} Son 5 Maç**")
            last5_a = predictor.trainer.get_last_n_matches(away_team)
            if not last5_a.empty:
                df_a = last5_a.copy()
                df_a['date'] = pd.to_datetime(df_a['date']).dt.strftime('%d/%m')
                st.dataframe(
                    df_a[['date', 'home_team', 'away_team', 'home_score', 'away_score']].rename(columns=col_rename),
                    hide_index=True)

elif not selected_code:
    st.info("👈 Analiz yapmak için soldan bir lig seçin.")
