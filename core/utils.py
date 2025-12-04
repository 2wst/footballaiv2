# core/utils.py

import re

def clean_team_name(name: str) -> str:
    """
    Bir takım adının başındaki ('eng ') veya sonundaki (' nl') ülke kodlarını temizler.
    Örnek: 'eng Arsenal' -> 'Arsenal', 'Ajax nl' -> 'Ajax'
    """
    if not isinstance(name, str):
        return name

    # 1. Başındaki üç harfli kodu ve boşluğu sil (örn: "eng Arsenal")
    name = re.sub(r'^[a-z]{3}\s', '', name.strip())

    # 2. Sonundaki boşluğu ve iki harfli kodu sil (örn: "Ajax nl")
    name = re.sub(r'\s[a-z]{2}$', '', name.strip())

    # 3. Başındaki iki harfli kodu ve boşluğu sil (örn: "es Villarreal")
    name = re.sub(r'^[a-z]{2}\s', '', name.strip())

    # 4. Sonundaki üç harfli kodu ve boşluğu sil (örn: "Arsenal eng")
    name = re.sub(r'\s[a-z]{3}$', '', name.strip())

    return name.strip()