import numpy as np
import pandas as pd

def parse_num(series):
    """Bersihkan kolom numerik: hapus %, spasi, tanda strip."""
    return (
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace('-', '0', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )

def to_pct_numeric(series):
    s = pd.to_numeric(series, errors='coerce')
    return np.where(s.between(0, 1.5), s * 100, s)
