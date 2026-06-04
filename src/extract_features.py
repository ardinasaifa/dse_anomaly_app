import pandas as pd
import numpy as np

from src.utils import parse_num, to_pct_numeric

def extract_main_features(df, brand):
    cols = df.columns.tolist()

    # Rename kolom compliance & incentive hasil merge header Excel
    rename_map = {}
    if 'Unnamed: 39' in cols:
        rename_map['Unnamed: 39'] = 'TOTAL_COMPLIANCE'
    if 'Unnamed: 40' in cols:
        rename_map['Unnamed: 40'] = 'COMPLIANCE_INDEX'
    if 'Unnamed: 41' in cols:
        rename_map['Unnamed: 41'] = 'BEFORE_COMPLIANCE'
    if 'Unnamed: 42' in cols:
        rename_map['Unnamed: 42'] = 'FINAL_AFTER_COMPLIANCE'

    df = df.rename(columns=rename_map)

    # update cols setelah rename
    cols = df.columns.tolist()

    # Normalisasi nama kolom ID & flag
    id_col   = 'DSE ID'    if 'DSE ID'    in cols else 'DSE IDE'
    flag_col = 'flag densus' if 'flag densus' in cols else 'Flag densus'

    # Score Sellin (IM3: 'Score Sellin', 3ID: 'Scorre Sellin')
    score_sellin_col = 'Score Sellin' if 'Score Sellin' in cols else 'Scorre Sellin'
    # Score EC (IM3: 'Score EC', 3ID: 'SCORE EC')
    score_ec_col = 'Score EC' if 'Score EC' in cols else 'SCORE EC'
    # Champions FWA (IM3: 'FWA', 3ID: 'RGUGA FWA')
    fwa_col = 'FWA' if 'FWA' in cols else 'RGUGA FWA'
    # Target sellin (IM3: 'TARGET SELLIN 3PCS', 3ID: 'TARGET SELLIN')
    target_sellin_col = 'TARGET SELLIN 3PCS' if 'TARGET SELLIN 3PCS' in cols else 'TARGET SELLIN'
    # SP SELLIN champions (IM3: 'SP SELLIN ', 3ID: 'SP SELLIN.1')
    sp_champ_col = 'SP SELLIN ' if 'SP SELLIN ' in cols else ('SP SELLIN.1' if 'SP SELLIN.1' in cols else 'SP SELLIN')

    out = pd.DataFrame()
    out['DSE_ID']        = df[id_col].astype(str).str.strip()
    out['BRAND_SRC']     = brand
    out['FLAG_DENSUS']   = parse_num(df[flag_col])
    out['PARTNER_NAME']  = df.get('PARTNER NAME', pd.Series([''] * len(df))).astype(str)
    out['MICRO_CLUSTER'] = df.get('MICRO CLUSTER', pd.Series([''] * len(df))).astype(str)
    out['BRANCH']        = df.get('BRANCH', pd.Series([''] * len(df))).astype(str)
    out['AREA']          = df.get('AREA', pd.Series([''] * len(df))).astype(str)
    out['REGION']        = df.get('REGION', pd.Series([''] * len(df))).astype(str)

    # ── SP Sellin KPI
    out['SP_TARGET']      = parse_num(df['TARGET'])
    out['SP_ACTUAL']      = parse_num(df['ACTUAL'])
    out['SP_ACH']         = to_pct_numeric(df['%ACH'])
    out['SP_UNIT_PRICE']  = parse_num(df['UNIT PRICE'])
    out['SP_INCENTIVE']   = parse_num(df['INCENTIVE'])

    # ── OSA KPI
    out['OSA_TARGET']     = parse_num(df['TARGET.1'])
    out['OSA_ACTUAL']     = parse_num(df['ACTUAL.1'])
    out['OSA_ACH']        = to_pct_numeric(df['%ACH.1'])
    out['OSA_UNIT_PRICE'] = parse_num(df['UNIT PRICE.1'])
    out['OSA_INCENTIVE']  = parse_num(df['INCENTIVE.1'])

    # ── RGUGA FWA KPI
    out['FWA_TARGET']     = parse_num(df['TARGET.2'])
    out['FWA_ACTUAL']     = parse_num(df['ACTUAL.2'])
    out['FWA_ACH']        = to_pct_numeric(df['%ACH.2'])
    out['FWA_UNIT_PRICE'] = parse_num(df['UNIT PRICE.2'])
    out['FWA_INCENTIVE']  = parse_num(df['INCENTIVE.2'])

    # ── Coverage outlet
    out['OUTLET_MAPPING']  = parse_num(df.get('OUTLET MAPPING', pd.Series([0]*len(df))))
    out['FLAG_TARGET']     = parse_num(df.get('FLAG TARGET', pd.Series([0]*len(df))))
    out['TARGET_SELLIN']   = parse_num(df.get(target_sellin_col, pd.Series([0]*len(df))))
    out['SP_SELLIN_ACTUAL']= parse_num(df.get('SP SELLIN', pd.Series([0]*len(df))))

    # ── Score per dimensi
    out['SCORE_SELLIN']    = parse_num(df.get(score_sellin_col, pd.Series([0]*len(df))))
    out['SCORE_EC']        = parse_num(df.get(score_ec_col, pd.Series([0]*len(df))))

    # ── Champions score (SP SELLIN, OSA, FWA, Total)
    # out['CHAMP_SP']        = parse_num(df.get(sp_champ_col, pd.Series([0]*len(df))))
    # out['CHAMP_OSA']       = parse_num(df.get('OSA', pd.Series([0]*len(df))))
    # out['CHAMP_FWA']       = parse_num(df.get(fwa_col, pd.Series([0]*len(df))))
    out['CHAMP_TOTAL']     = parse_num(df.get('Total Score', pd.Series([0]*len(df))))

    # Rata-rata achievement 3 KPI
    out['KPI_AVG_ACH']     = out[['SP_ACH','OSA_ACH','FWA_ACH']].mean(axis=1)

    # Konsistensi KPI (std rendah = seimbang antar KPI)
    out['KPI_STD']         = out[['SP_ACH','OSA_ACH','FWA_ACH']].std(axis=1)

    # Compliance
    out['COMPLIANCE_INDEX'] = parse_num(df.get('COMPLIANCE_INDEX', pd.Series([0]*len(df))))

    # Total insentif
    out['TOTAL_INCENTIVE'] = out['SP_INCENTIVE'] + out['OSA_INCENTIVE'] + out['FWA_INCENTIVE']
    out['INCENTIVE_FINAL_AFTER_COMPLIANCE'] =  parse_num(df.get('FINAL_AFTER_COMPLIANCE', pd.Series([0]*len(df))))

    # Coverage rate (SP SELLIN actual / TARGET SELLIN)
    out['COVERAGE_RATE'] = out['SP_SELLIN_ACTUAL'] / (out['TARGET_SELLIN'].replace(0, np.nan))

    return out.dropna(subset=['DSE_ID'])



def extract_ec_features(df, brand):
    """
    Ekstrak fitur dari pola harian Visit / Sellin>50k / EC.
    Handle perbedaan IM3 vs 3ID:
    - IM3: threshold col = 'THEREHOLD EC', outlet = 'OUTLET MAPPING', ada 1 kolom ' ' (hari libur day-5)
    - 3ID: threshold col = 'THERSHOLD EC', outlet = 'OUTLET PJP'
    """
    cols = df.columns.tolist()

    id_col        = 'DSE ID'
    thresh_col    = 'THEREHOLD EC' if 'THEREHOLD EC' in cols else 'THERSHOLD EC' if 'THERSHOLD EC' in cols else 'THRESHOLD EC'
    outlet_col    = 'OUTLET MAPPING' if 'OUTLET MAPPING' in cols else 'OUTLET PJP'

    df = df.dropna(subset=[id_col]).copy() # Pastikan hanya DSE yang memiliki ID yang diproses
    df = df.rename(columns={id_col: 'DSE_ID'})

    # Identifikasi kelompok Visit/Sellin/EC
    # Pola: Visit, Sellin>50k, EC berulang untuk 30 hari
    visit_cols  = [c for c in cols if str(c).startswith('Visit')]
    sellin_cols = [c for c in cols if str(c).startswith('Sellin')]
    ec_cols     = [c for c in cols if str(c).startswith('EC') and c not in ['EC.x','EC.y']]

    n_days = min(len(visit_cols), len(sellin_cols), len(ec_cols))
    print(f'  [{brand}] Hari terdeteksi: {n_days}')

    rows = []
    for _, row in df.iterrows():
        dse_id    = str(row['DSE_ID']).strip()
        threshold = pd.to_numeric(str(row.get(thresh_col, 0)).replace(' ',''), errors='coerce') or 0
        outlet    = pd.to_numeric(str(row.get(outlet_col, 0)).replace(' ',''), errors='coerce') or 0

        visits, sellins, ecs = [], [], []
        for i in range(n_days):
            v  = pd.to_numeric(str(row.get(visit_cols[i],  0)).replace(' ','').replace('-','0'), errors='coerce') or 0
            s  = pd.to_numeric(str(row.get(sellin_cols[i], 0)).replace(' ','').replace('-','0'), errors='coerce') or 0
            e  = str(row.get(ec_cols[i], '0%'))
            en = pd.to_numeric(e.replace('%','').replace(' ',''), errors='coerce') or 0
            if en <= 1.5: en *= 100
            visits.append(v); sellins.append(s); ecs.append(en)

        # Hari libur = visit=0 & sellin=0
        hari_libur = [v == 0 and s == 0 for v, s in zip(visits, sellins)]
        kerja_idx  = [i for i, h in enumerate(hari_libur) if not h]
        n_kerja    = len(kerja_idx) or 1

        vk = [visits[i]  for i in kerja_idx]
        sk = [sellins[i] for i in kerja_idx]
        ek = [ecs[i]     for i in kerja_idx]

        # ── Fitur agregat
        avg_visit   = np.mean(vk) if vk else 0
        avg_sellin  = np.mean(sk) if sk else 0
        ec_rate     = np.mean([1 if e==100 else 0 for e in ek]) if ek else 0
        visit_std   = np.std(vk) if len(vk)>1 else 0 # Untuk melihat konsistensi kunjungan harian, std kecil = konsisten
        # Coefficient of Variation (CV) untuk visit & sellin
        visit_cv = visit_std / avg_visit if avg_visit > 0 else 0 # Untuk melihat konsistensi kunjungan harian relatif terhadap rata-rata, cv kecil = konsisten
        sellin_cv = np.std(sk) / avg_sellin if avg_sellin > 0 else 0 # Untuk melihat konsistensi penjualan relatif terhadap rata-rata, cv kecil = konsisten

        # Produktivitas: outlet yang di-visit dengan sellin per hari
        sellin_per_visit = avg_sellin / avg_visit if avg_visit > 0 else 0

        # ── Konsentrasi waktu kunjungan
        # EOM: % visit di 5 hari terakhir
        last5 = kerja_idx[-5:] if len(kerja_idx)>=5 else kerja_idx
        eom_visit = sum(visits[i] for i in last5)
        total_visit = sum(vk) or 1
        eom_concentration = eom_visit / total_visit

        # SOM: % visit di 5 hari pertama
        first5 = kerja_idx[:5] if len(kerja_idx)>=5 else kerja_idx
        som_visit = sum(visits[i] for i in first5)
        som_concentration = som_visit / total_visit

        # Rasio kunjungan tengah bulan (hari ke 8-22)
        mid_idx = [i for i in kerja_idx if 7 <= i <= 21]
        mid_visit = sum(visits[i] for i in mid_idx)
        mid_concentration = mid_visit / total_visit

        # ── Threshold compliance
        below_threshold_rate = np.mean([1 if v < threshold else 0 for v in vk]) if vk else 0

        # ── Tren EC mingguan (apakah membaik atau memburuk)
        # Bagi 4 minggu, hitung rata-rata EC per minggu
        week_ec = []
        for w in range(4):
            w_idx = [i for i in kerja_idx if w*7 <= i < (w+1)*7]
            if w_idx:
                week_ec.append(np.mean([ecs[i] for i in w_idx]))
        ec_trend = 0
        if len(week_ec) >= 2:
            # Slope sederhana: positif = membaik, negatif = memburuk
            ec_trend = (week_ec[-1] - week_ec[0]) / (len(week_ec) - 1)

        # ── Aggregat summary dari kolom ringkasan
        jumlah_visit = sum(visits)
        total_sellin = sum(sellins)
        avg_ec_summary = pd.to_numeric(str(row.get('ec', 0)).replace('%','').replace(' ',''), errors='coerce') or ec_rate
        if avg_ec_summary <= 1.5: avg_ec_summary *= 100

        rows.append({
            'DSE_ID'               : dse_id,
            'BRAND_SRC'            : brand,
            'OUTLET    '           : outlet,
            'THRESHOLD_EC'         : threshold,
            'N_HARI_KERJA'         : n_kerja,
            # Visit
            'AVG_VISIT'            : round(avg_visit, 2),
            'VISIT_CV'             : round(visit_cv, 3),
            # Sellin
            'AVG_SELLIN'           : round(avg_sellin, 2),
            'SELLIN_CV'            : round(sellin_cv, 3),
            'SELLIN_PER_VISIT'     : round(sellin_per_visit, 2),
            # EC
            'EC_RATE'              : round(ec_rate, 3),
            'EC_TREND_WEEKLY'      : round(ec_trend, 2),
            # Konsentrasi waktu
            'EOM_CONCENTRATION'    : round(eom_concentration, 3),
            'SOM_CONCENTRATION'    : round(som_concentration, 3),
            'MID_CONCENTRATION'    : round(mid_concentration, 3),
            # Threshold
            'BELOW_THRESHOLD_RATE' : round(below_threshold_rate, 3),
            # Summary
            'TOTAL_SELLIN_SUMMARY' : total_sellin,
            'TOTAL_VISIT_SUMMARY'  : jumlah_visit,
            'AVG_EC_SUMMARY'       : round(avg_ec_summary, 1),
        })

    return pd.DataFrame(rows)


# def merge_features(df_main_feat, df_ec_feat):

#     df_feat = df_ec_feat.merge(
#         df_main_feat.drop(columns=['BRAND_SRC'],
#         errors='ignore'),
#         on='DSE_ID',
#         how='left'
#     )

#     return df_feat