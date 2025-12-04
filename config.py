# config.py

DATA_FOLDER = "data"
MODELS_FOLDER = "models"

# Lig listesini football-data.co.uk'de bulunan kodlarla güncelledik.
# 'CL' kaldırıldı çünkü bu sitede standart lig formatında bulunmuyor.
# 'INT' yerel dosyadan okunmaya devam edecek.
LEAGUE_CODES = ['D1', 'E0', 'F1', 'I1', 'INT', 'N1', 'P1', 'SC0', 'SP1', 'T1']

# --- ELO AYARLARI ---
# Lig ağırlıklarını mevcut haliyle koruyoruz, yeni verilerle de anlamlı olacaktır.
LEAGUE_WEIGHTS = {
    'CL': 1.10, 'E0': 1.00, 'I1': 0.87, 'SP1': 0.86, 'D1': 0.83, 'F1': 0.64, 'N1': 0.59,
    'P1': 0.54, 'T1': 0.37, 'SC0': 0.35, 'INT': 0.90
}
INITIAL_ELO = 1500
K_FACTOR = 30

# --- TARİH KISITLAMA AYARI (BACKTEST) ---
# Eğer geçmişe dönük test yapacaksanız buraya tarih yazın (Örn: "2025-11-30").
# Eğer GÜNCEL tahmin yapacaksanız ve tüm veriyi istiyorsanız burayı None yapın.

#CUTOFF_DATE = "2025-11-30"  # Açık: Sadece 30 Kasım'a kadar olan veriyi alır.
CUTOFF_DATE = None        # Kapalı: Tüm güncel verileri alır.