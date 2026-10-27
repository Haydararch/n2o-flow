# -*- coding: utf-8 -*-
"""
GAZAP N2O Enjektör Akış Hesabı - Streamlit arayüzü (V7.3)
Hesap motoru: n2o_enjektor_v7_3.py (aynı klasörde olmalı)
Yerelde çalıştırma:  streamlit run app.py
"""
import contextlib
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import n2o_enjektor_v7_3 as m

st.set_page_config(page_title="GAZAP N2O Enjektör", page_icon="🔥", layout="wide")

# Streamlit sürüm uyumu: yeni sürümler width='stretch', eski sürümler use_container_width=True ister
try:
    _sv = tuple(int(x) for x in st.__version__.split('.')[:2])
except ValueError:
    _sv = (1, 0)
GENIS = {'width': 'stretch'} if _sv >= (1, 46) else {'use_container_width': True}

MALZEMELER = ['AISI_316L', 'AISI_304', 'Inconel_718', 'Al_6061_T6', 'CuCrZr']
MODEL_KISA = {
    'spi': 'SPI', 'spc_liq': 'SPC sıvı', 'hem_lokal': 'HEM yerel', 'nurick': 'Nurick',
    'hem_c': 'HEMc', 'dyer': 'Dyer', 'dyer_comp': 'Dyer-komp.', 'fml': 'FML', 'dual_spc': 'Dual-SPC',
    'omega': 'Leung ω',
}


def _f(v, fmt="{:.3f}"):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


# --------------------------------------------------------------------------- hesap
@st.cache_data(show_spinner=False)
def _vv():
    return m.vv_makale_karsilastirmasi()


@st.cache_data(show_spinner=False, max_entries=32)
def hesapla(g: dict, kal_satirlari: tuple):
    op = m.OperatingConfig(
        m_dot_rpa=g['m_dot_rpa'], eta_atomizasyon=g['eta'],
        P_tank_pa=g['P_tank_bar'] * 1e5, T_tank_K=g['T_tank_C'] + 273.15, P2_pa=g['P2_bar'] * 1e5,
        exponent_mode=g['exponent_mode'], vap_ref_mode=g['vap_ref_mode'], slip_mode=g['slip_mode'],
        tasarim_modeli=g['tasarim_modeli'],
    )
    mf = None
    if g['geometri'] == 'orifis':
        orif = m.OrificeConfig(
            geometri='orifis', Cd_tekfaz=g['Cd'], N_orifis=g['N'], L_orifis_mm=g['L'],
            inlet_edge_type=g['edge'], Cc_override=g['Cc_override'], drill_tolerance_mm=g['tol'])
    else:
        orif = m.OrificeConfig(
            geometri='annulus', Cd_tekfaz=g['Cd'], D_ic_mm=g['D_ic'], L_orifis_mm=g['L'],
            inlet_edge_type=g['edge'], Cc_override=g['Cc_override'], drill_tolerance_mm=g['tol'],
            eksantriklik_mm=g['eks'])
        mf = m.ManifoldConfig(D_ic_mm=g['mf_Di'], D_dis_mm=g['mf_Do'], L_mm=g['mf_L'], N_giris=int(g['mf_N']),
                              d_giris_mm=g['mf_d'], K_giris=g['mf_Kg'], K_donus=g['mf_Kd'], sigma_i=g['mf_sig'])
    feed = m.FeedSystemConfig(
        D_hat_mm=g['D_hat'], L_hat_m=g['L_hat'], eps_boru_mm=g['eps'], K_tank_cikis=g['K_tank'],
        Cv_ana_vana=g['Cv'], FL_vana=g['FL'], K_fitting_toplam=g['K_fit'], K_manifold_giris=g['K_man'],
        D_manifold_ic_mm=g['D_man'], H_manifold_ic_mm=g['H_man'], manifold=mf)
    th = m.ThermalConfig(
        T_gaz_eff_low_K=g['Tg_lo'], T_gaz_eff_nom_K=g['Tg_nom'], T_gaz_eff_high_K=g['Tg_hi'],
        h_gaz_low=g['hg_lo'], h_gaz_nominal=g['hg_nom'], h_gaz_high=g['hg_hi'],
        D_ayna_mm=g['D_ayna'], D_merkez_mm=g['D_merkez'], malzeme=g['malzeme'], t_yanma_s=g['t_yanma'],
        t_kapanma_sonrasi_s=g['t_kap'], h_n2o_durgun=g['h_durgun'], V_hapsolan_cm3=g['V_hap'],
        f_mix_nominal=g['f_mix'], T_wall_caution_K=g['T_uyari_C'] + 273.15,
        T_wall_hazard_K=g['T_tehlike_C'] + 273.15)

    testler = [m.KalibrasyonTesti(P1_bar=r[0], T1_K=r[1] + 273.15, P2_bar=r[2], d_mm=r[3],
                                  N_orifis=int(r[4]), mdot_olculen=r[5], D_ic_mm=(r[6] or None))
               for r in kal_satirlari]
    kal_ozet = m.kalibrasyon_hesapla(testler, op, orif) if testler else {}
    kal = {k: v[0] for k, v in kal_ozet.items()}

    vv = _vv()
    sol = m.baglasik_coz(op, orif, feed, th, kal)
    boyutlar = [round(float(x), 3) for x in np.arange(g['b_min'], g['b_max'] + 1e-9, g['b_adim'])]
    tablo = m.matkap_tablosu_coz(sol, op, orif, feed, kal, boyutlar)
    uq = m.belirsizlik_coz(sol, op, orif, kal, n=g['n_mc'])

    d_sabit = g['d_sabit'] if g['d_sabit'] > 0 else sol.boy.d_tasarim_mm
    man = (m.manifold_coz(op.m_dot_hedef, sol.st_in, mf, orif, d_sabit, sol.fl, op.P2_pa) if mf is not None else None)
    T_liste = [round(float(x), 2) for x in np.arange(g['T_min'], g['T_max'] + 1e-9, g['T_adim'])]
    tarama = m.sicaklik_taramasi(op, orif, feed, kal, d_sabit, T_liste, g['OF_rpa'])
    sen = None
    if mf is not None:
        x_liste = tuple(round(float(x), 4) for x in np.arange(0.0, g['x_max'] + 1e-9, g['x_adim']))
        sen = m.senaryo_coz(op, orif, feed, kal, sol, d_sabit, g['OF_rpa'], g['OF_bant'], g['x_bilinen'], x_liste)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.rapor_yazdir(op, orif, feed, th, vv, sol, tablo, uq, kal_ozet, vv_yazdir=True,
                       man=man, tarama=tarama, senaryolar=sen)
    return dict(op=op, orif=orif, th=th, feed=feed, sol=sol, tablo=tablo, uq=uq, kal_ozet=kal_ozet, vv=vv,
                man=man, tarama=tarama, sen=sen, d_sabit=d_sabit, rapor=buf.getvalue())


# --------------------------------------------------------------------------- girdiler
st.sidebar.title("Girdiler")
geometri = st.sidebar.radio("Çıkış geometrisi", ['annulus', 'orifis'], horizontal=True,
                            format_func=lambda s: 'Orifis (N delik)' if s == 'orifis' else 'Annulus (halka)')

with st.sidebar.form("girdiler"):
    g = {'geometri': geometri}
    with st.expander("İşletme", expanded=True):
        g['m_dot_rpa'] = st.number_input("RPA N2O debisi [kg/s]", 0.01, 20.0, 1.09608, 0.001, format="%.5f")
        g['eta'] = st.number_input("Atomizasyon verimi η [-]", 0.30, 1.00, 0.90, 0.01,
                                   help="Hedef debi = RPA debisi / η")
        g['P_tank_bar'] = st.number_input("Tank basıncı [bar]", 5.0, 72.0, 62.0, 0.5,
                                          help="Helyumla basınçlandırma (regülatörlü, sıcaklıktan bağımsız sabit kabul)")
        g['T_tank_C'] = st.number_input("Tank sıvı sıcaklığı [°C]", -30.0, 36.0, 20.0, 0.5)
        g['P2_bar'] = st.number_input("Oda basıncı P2 [bar]", 1.0, 70.0, 27.0, 0.5, help="Kesinleşti: 27 bar")
        c1, c2 = st.columns(2)
        g['OF_rpa'] = c1.number_input("RPA O/F", 1.0, 12.0, 4.0, 0.1,
                                      help="Yakıt debisi sabit; hedef N2O debisinde O/F bu değere eşit kabul edilir")
        g['OF_bant'] = c2.number_input("O/F kabul ±%", 1.0, 50.0, 10.0, 1.0)

    with st.expander("Çıkış kanalı", expanded=True):
        if geometri == 'orifis':
            g['N'] = st.number_input("Orifis sayısı N", 1, 200, 8, 1)
            g['D_ic'], g['eks'] = None, 0.0
            g['Cd'] = st.number_input("Cd tek faz [-]", 0.31, 1.00, 0.60, 0.01,
                                      help="Kavitasyonsuz tek faz Cd. Keskin girişli kısa delik (L/d 2-8) tipik 0.75-0.82, "
                                           "ince levha orifis 0.61. Kavitasyon ve flaş modellerde ayrıca hesaplanır.")
            b_min, b_max, b_adim = 2.00, 2.95, 0.05
            lbl = "çap"
        else:
            g['N'] = 1
            g['D_ic'] = st.number_input("Halka iç çapı [mm]", 1.0, 200.0, 14.0, 0.5, help="GAZAP: pintle dış çapı Ø14")
            g['eks'] = st.number_input("Eksantriklik [mm]", 0.0, 1.0, 0.02, 0.005, format="%.3f")
            g['Cd'] = st.number_input("Cd tek faz [-]", 0.31, 1.00, 0.75, 0.01,
                                      help="Kavitasyonsuz tek faz Cd. İnce aralıkta sürtünmeyle ~0.75-0.8.")
            b_min, b_max, b_adim = 0.60, 1.10, 0.05
            lbl = "aralık"
        g['L'] = st.number_input("Kanal boyu L [mm]", 0.5, 50.0, 8.0, 0.5)
        g['edge'] = st.selectbox("Giriş kenarı", ['sharp', 'chamfered', 'rounded'],
                                 format_func={'sharp': 'Keskin (Cc 0.61)', 'chamfered': 'Pahlı (Cc 0.68)',
                                              'rounded': 'Yuvarlatılmış (Cc 0.80)'}.get)
        cc_var = st.checkbox("Ölçülmüş Cc gir", False)
        g['Cc_override'] = st.number_input("Cc [-]", 0.30, 1.00, 0.61, 0.01) if cc_var else None
        g['tol'] = st.number_input(f"İşleme toleransı ({lbl}) [±mm]", 0.0, 0.5, 0.02, 0.005, format="%.3f")
        c1, c2, c3 = st.columns(3)
        g['b_min'] = c1.number_input(f"Tablo min {lbl}", 0.01, 50.0, b_min, 0.05)
        g['b_max'] = c2.number_input(f"Tablo max {lbl}", 0.01, 50.0, b_max, 0.05)
        g['b_adim'] = c3.number_input("Adım", 0.005, 5.0, b_adim, 0.005, format="%.3f")

    if geometri == 'annulus':
        with st.expander("Manifold (annulus)", expanded=True):
            c1, c2, c3 = st.columns(3)
            g['mf_Di'] = c1.number_input("İç çap [mm]", 1.0, 300.0, 14.0, 0.5)
            g['mf_Do'] = c2.number_input("Dış çap [mm]", 2.0, 300.0, 44.0, 0.5)
            g['mf_L'] = c3.number_input("Boy [mm]", 0.5, 100.0, 8.0, 0.5, help="Eksenel derinlik")
            c1, c2 = st.columns(2)
            g['mf_N'] = c1.number_input("Giriş portu sayısı", 1, 12, 2, 1)
            g['mf_d'] = c2.number_input("Port çapı [mm]", 1.0, 50.0, 10.0, 0.5,
                                        help="Toplam port alanı hat alanına yakın olmalı (1/2\" hat ≈ 194 mm²)")
            c1, c2, c3 = st.columns(3)
            g['mf_Kg'] = c1.number_input("K port", 0.0, 5.0, 1.0, 0.1, help="Port çıkışı ani genişleme")
            g['mf_Kd'] = c2.number_input("K dönüş", 0.0, 5.0, 0.5, 0.1, help="Eksenel akıştan annulus girişine dönüş")
            g['mf_sig'] = c3.number_input("σ_i", 0.5, 5.0, 2.0, 0.1, help="Port jeti kavitasyon başlangıç sayısı")
    else:
        g.update(mf_Di=0.0, mf_Do=0.0, mf_L=0.0, mf_N=1, mf_d=0.0, mf_Kg=0.0, mf_Kd=0.0, mf_sig=0.0)

    with st.expander("Besleme hattı"):
        g['D_hat'] = st.number_input("Hat iç çapı [mm]", 2.0, 100.0, 15.7, 0.1)
        g['L_hat'] = st.number_input("Hat uzunluğu [m]", 0.05, 20.0, 1.0, 0.05)
        g['eps'] = st.number_input("Pürüzlülük [mm]", 0.0, 1.0, 0.015, 0.001, format="%.4f")
        g['K_tank'] = st.number_input("K tank çıkışı [-]", 0.0, 10.0, 0.5, 0.1)
        g['Cv'] = st.number_input("Ana vana Cv", 0.1, 100.0, 5.5, 0.1)
        g['FL'] = st.number_input("Vana FL [-]", 0.3, 1.0, 0.60, 0.01)
        g['K_fit'] = st.number_input("K dirsek/rakor toplam [-]", 0.0, 50.0, 0.8, 0.1)
        if geometri == 'orifis':
            g['K_man'] = st.number_input("K manifold girişi [-]", 0.0, 20.0, 1.5, 0.1)
            g['D_man'] = st.number_input("Galeri çapı [mm]", 5.0, 300.0, 60.0, 1.0)
            g['H_man'] = st.number_input("Galeri yüksekliği [mm]", 0.5, 100.0, 10.0, 0.5)
        else:
            g['K_man'], g['D_man'], g['H_man'] = 1.5, 60.0, 10.0

    with st.expander("Sıcaklık taraması"):
        c1, c2, c3 = st.columns(3)
        g['T_min'] = c1.number_input("T min [°C]", -20.0, 40.0, 0.0, 1.0)
        g['T_max'] = c2.number_input("T max [°C]", -20.0, 40.0, 40.0, 1.0)
        g['T_adim'] = c3.number_input("Adım [°C]", 0.5, 10.0, 2.5, 0.5)
        g['d_sabit'] = st.number_input(f"Sabit {lbl} [mm] (0 = tasarım modeli)", 0.0, 50.0, 0.0, 0.01, format="%.3f",
                                       help="Sıcaklık tablosu ve senaryolar bu boyutla hesaplanır")

    if geometri == 'annulus':
        with st.expander("Senaryolar (S3 / S4)"):
            g['x_bilinen'] = st.number_input("S3 giriş kuruluğu x1 [%]", 0.0, 50.0, 5.0, 0.5) / 100.0
            c1, c2 = st.columns(2)
            g['x_max'] = c1.number_input("S4 x1 max [%]", 1.0, 50.0, 20.0, 1.0) / 100.0
            g['x_adim'] = c2.number_input("S4 x1 adım [%]", 0.5, 10.0, 2.5, 0.5) / 100.0
    else:
        g.update(x_bilinen=0.05, x_max=0.2, x_adim=0.025)

    with st.expander("Ayna / termal"):
        g['malzeme'] = st.selectbox("Ayna malzemesi", MALZEMELER)
        g['t_yanma'] = st.number_input("Yanma süresi [s]", 0.5, 60.0, 5.0, 0.5)
        g['D_ayna'] = st.number_input("Ayna dış çapı [mm]", 5.0, 300.0, 83.4, 0.1)
        g['D_merkez'] = st.number_input("Merkez yakıt gövdesi çapı [mm]", 0.0, 200.0, 20.0, 0.5)
        g['V_hap'] = st.number_input("Vana-ayna arası hapsolan hacim [cm³]", 0.1, 2000.0, 25.0, 1.0)
        g['h_durgun'] = st.number_input("Durgun N2O h [W/m²K]", 10.0, 5000.0, 300.0, 10.0)
        g['t_kap'] = st.number_input("Kapanma sonrası izleme [s]", 1.0, 600.0, 30.0, 1.0)
        g['f_mix'] = st.number_input("Isının bulk akışa karışma oranı [-]", 0.0, 1.0, 0.35, 0.05)
        st.caption("Gaz tarafı sınır şartı zarfı (RPA / Bartz)")
        a, b_, c = st.columns(3)
        g['Tg_lo'] = a.number_input("T_gaz düşük [K]", 300.0, 4000.0, 1400.0, 50.0)
        g['Tg_nom'] = b_.number_input("T_gaz nom [K]", 300.0, 4000.0, 2050.0, 50.0)
        g['Tg_hi'] = c.number_input("T_gaz yüksek [K]", 300.0, 4000.0, 2453.0, 50.0)
        g['hg_lo'] = a.number_input("h_gaz düşük", 10.0, 20000.0, 500.0, 50.0)
        g['hg_nom'] = b_.number_input("h_gaz nom", 10.0, 20000.0, 950.0, 50.0)
        g['hg_hi'] = c.number_input("h_gaz yüksek", 10.0, 20000.0, 1600.0, 50.0)
        g['T_uyari_C'] = a.number_input("Uyarı eşiği [°C]", 50.0, 1000.0, 200.0, 10.0)
        g['T_tehlike_C'] = b_.number_input("Tehlike eşiği [°C]", 50.0, 1000.0, 300.0, 10.0)

    with st.expander("Model ayarları"):
        g['tasarim_modeli'] = st.selectbox("Tasarım (referans) modeli", list(m.BANT_MODELLERI) +
                                           ['spi', 'spc_liq', 'hem_lokal', 'dyer'],
                                           index=list(m.BANT_MODELLERI).index('dual_spc'),
                                           format_func=lambda k: MODEL_KISA[k])
        g['exponent_mode'] = st.selectbox("İzentropik üs", ['real_ns', 'cp_cv'])
        g['vap_ref_mode'] = st.selectbox("Buhar referansı", ['T1_sat', 'P1_sat'])
        g['slip_mode'] = st.selectbox("Kayma oranı", ['zivi', 'homogeneous'])
        g['n_mc'] = st.select_slider("Monte Carlo örnek sayısı", [500, 1000, 2000, 4000, 8000], 4000)

    with st.expander("Kalibrasyon testleri"):
        st.caption("Kendi ölçümlerin. Annulus testinde: d = aralık, N = 1, D_iç dolu. Orifiste D_iç boş.")
        kal_df = st.data_editor(
            pd.DataFrame(columns=['P1 [bar]', 'T1 [°C]', 'P2 [bar]', 'd [mm]', 'N', 'ṁ [kg/s]', 'D_iç [mm]']),
            num_rows="dynamic", **GENIS, key="kal")

    gonder = st.form_submit_button("Hesapla", type="primary", **GENIS)

kal_satirlari = tuple(
    tuple(None if pd.isna(v) else float(v) for v in row)
    for row in kal_df.itertuples(index=False)
    if not any(pd.isna(row[i]) for i in range(6))
)

# --------------------------------------------------------------------------- başlık
st.title("GAZAP · N2O Enjektör Akış Hesabı")
st.caption("Alaz Space · hesap motoru V7.3 · yalnızca N2O tarafı. "
           "Sonuç tek bir değer değil, modeller arası bant: kanalı bandın küçük ucundan işleyip test sonrası büyütün.")

if gonder or 'sonuc' not in st.session_state:
    try:
        with st.spinner("Hesaplanıyor (sıcaklık taraması ve senaryolarla ilk çalıştırma ~40-60 s sürebilir)..."):
            st.session_state['sonuc'] = hesapla(g, kal_satirlari)
            st.session_state['hata'] = None
    except ValueError as e:
        st.session_state['hata'] = str(e)

if st.session_state.get('hata'):
    st.error(st.session_state['hata'])
    st.stop()

R = st.session_state['sonuc']
op, orif, th, sol, tablo, uq = R['op'], R['orif'], R['th'], R['sol'], R['tablo'], R['uq']
man, tarama, sen, d_sabit = R['man'], R['tarama'], R['sen'], R['d_sabit']
fl, b, fd, t, si = sol.fl, sol.boy, sol.fd, sol.th, sol.st_in
bad = 'aralık' if orif.annulus else 'çap'
geo = f"Annulus, iç çap {orif.D_ic_mm:.2f} mm" if orif.annulus else f"Orifis, N = {orif.N_orifis}"
OF_rpa, OF_bant = g['OF_rpa'], g['OF_bant']

# --------------------------------------------------------------------------- özet
st.subheader(f"Özet · {geo} · hedef {op.m_dot_hedef:.4f} kg/s · P2 {op.P2_pa/1e5:.1f} bar")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"{bad.capitalize()} bandı [mm]", f"{b.bant_min_mm:.2f}–{b.bant_max_mm:.2f}",
          help=f"{MODEL_KISA[b.bant_min_model]} .. {MODEL_KISA[b.bant_max_model]}")
k2.metric("İlk işleme [mm]", f"{b.ilk_delme_mm:.2f}",
          help=(f"Dış çap {orif.D_dis_mm(b.ilk_delme_mm):.3f} mm" if orif.annulus else "Bandın küçük ucu"))
k3.metric(f"Tasarım modeli {bad} [mm]", f"{b.d_tasarim_mm:.3f}", help=MODEL_KISA[op.tasarim_modeli])
k4.metric("Manifold basıncı [bar]", f"{fd.P1_efektif_pa/1e5:.2f}", f"-{fd.dP_toplam_pa/1e5:.2f} hat kaybı",
          delta_color="off")
k5.metric("Maks. tank sıcaklığı [°C]", f"{fd.T_tank_max_K-273.15:.1f}",
          f"şu an {op.T_tank_K-273.15:.1f}", delta_color="off", help="Manifoldda kaynama başlamadan")
k6.metric("Kapanma sonrası ayna [°C]", f"{t.T_denge_kuru_nom_K-273.15:.0f} / {t.T_denge_kuru_K-273.15:.0f}",
          help="Nominal / en kötü zarf")

uyarilar = list(dict.fromkeys(list(si.uyarilar) + fl.uyarilar + b.uyarilar + fd.uyarilar + (man.uyarilar if man else [])))
durum = t.durum_ozeti
(st.error if durum.startswith("KRİTİK") else st.warning if durum.startswith(("UYARI", "SOAKBACK")) else st.info)(
    "Termal: " + durum)
if uyarilar:
    with st.expander(f"Uyarılar ({len(uyarilar)})", expanded=False):
        for w in uyarilar:
            st.write("• " + w)

sekme_adlari = ["Modeller", f"{bad.capitalize()} tablosu", "Manifold / besleme", "Sıcaklık"]
if sen:
    sekme_adlari.append("Senaryolar")
sekme_adlari += ["Termal", "Belirsizlik", "Makale kontrolü", "Tam rapor"]
sekmeler = dict(zip(sekme_adlari, st.tabs(sekme_adlari)))

# --------------------------------------------------------------------------- modeller
with sekmeler["Modeller"]:
    rows = []
    for k in m.MODEL_ADLARI:
        rol = "BANT" if k in m.BANT_MODELLERI else "referans"
        if k == op.tasarim_modeli:
            rol += " + TASARIM"
        rows.append({"Model": m.MODEL_ADLARI[k], "Rol": rol, "G [kg/m²s]": round(fl.G_kal[k]),
                     "Alan [mm²]": round(b.A_mm2[k], 2), f"{bad} [mm]": round(b.d_mm[k], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **GENIS)
    c1, c2 = st.columns(2)
    c1.markdown(
        f"**Akış rejimi**  \n"
        f"Nurick K = {fl.K_nurick:.3f}, K_kr = {fl.K_kritik:.2f} → "
        f"{'kavitasyonlu' if fl.kavitasyonlu else 'kavitasyonsuz'} (Cc {fl.Cc:.2f})  \n"
        f"HEM boğulma basıncı {fl.P_hem_bogulma_pa/1e5:.2f} bar → {'boğulmuş' if fl.hem_bogulmus else 'boğulmamış'}  \n"
        f"Çıkış kuruluğu %{fl.x_2*100:.1f}, void %{fl.alpha_2*100:.1f}, Dual-SPC Mach {fl.Mach_dual_homojen:.2f}  \n"
        f"L/D_h = {b.L_d:.2f}, Re = {b.Re_delik:.2e}, kalma süresi {fl.tau_r_s*1e6:.0f} µs")
    c2.markdown(
        f"**Kanal girişi**  \n"
        f"{si.P_pa/1e5:.2f} bar, {si.T_K-273.15:.2f} °C, alt-soğutma {si.alt_sogutma_pa/1e5:.2f} bar  \n"
        f"ρ_l = {si.rho_liq:.1f} kg/m³, Pv(T1) = {si.Pv_pa/1e5:.2f} bar  \n"
        f"**Jet hızı / momentum (TMR için)**  \n"
        f"{fl.v_jet_sivi:.0f} – {fl.v_jet_hem:.0f} m/s → {op.m_dot_hedef*fl.v_jet_sivi:.0f} – "
        f"{op.m_dot_hedef*fl.v_jet_hem:.0f} N")
    st.caption("Referans: banda girmez (SPI/SPC fiziksel üst sınır, HEM yerel aşırı kötümser, orijinal Dyer makalede en kötü). "
               "BANT: bu koşulda savunulabilir beş model. TASARIM: termal, tornado ve tolerans için kullanılan tek boyut.")

# --------------------------------------------------------------------------- boyut tablosu
with sekmeler[f"{bad.capitalize()} tablosu"]:
    kol = ['spi'] + list(m.BANT_MODELLERI)
    df = pd.DataFrame([{bad + " [mm]": r.d_mm, **{MODEL_KISA[k]: r.mdot[k] for k in kol},
                        "±tol (tasarım)": f"{r.mdot_tol_eksi:.3f}–{r.mdot_tol_arti:.3f}",
                        "dlnṁ/dlnP2": r.dlnm_dlnP2, "Not": r.not_str} for r in tablo])
    uzun = pd.DataFrame([{"boyut": r.d_mm, "Model": MODEL_KISA[k], "mdot": r.mdot[k]}
                         for r in tablo for k in kol]).dropna()
    cizgi = alt.Chart(uzun).mark_line(point=True).encode(
        x=alt.X("boyut:Q", title=f"{bad} [mm]", scale=alt.Scale(zero=False)),
        y=alt.Y("mdot:Q", title="ṁ [kg/s]"),
        color=alt.Color("Model:N", sort=[MODEL_KISA[k] for k in kol]),
        tooltip=["Model", alt.Tooltip("boyut:Q", title=f"{bad} [mm]"), alt.Tooltip("mdot:Q", title="ṁ [kg/s]", format=".3f")])
    hedef = alt.Chart(pd.DataFrame({"hedef": [op.m_dot_hedef]})).mark_rule(color="#d62728", strokeDash=[6, 4], strokeWidth=2).encode(y="hedef:Q")
    st.altair_chart((cizgi + hedef).properties(height=380), **GENIS)
    st.caption(f"Kırmızı kesikli çizgi: hedef debi {op.m_dot_hedef:.3f} kg/s. SPI referanstır (banda girmez). "
               "Bir boyuttaki modeller arası yayılma = o boyutla beklenen debi belirsizliği.")
    st.dataframe(df.round(3), hide_index=True, **GENIS)

# --------------------------------------------------------------------------- manifold / besleme
with sekmeler["Manifold / besleme"]:
    zincir = pd.DataFrame([
        ("Tank", op.P_tank_pa / 1e5, 0.0),
        ("Tank çıkışı", fd.P_vana_oncesi_pa / 1e5, fd.dP_tank_cikis_pa / 1e5),
        ("Vana sonrası", fd.P_vana_sonrasi_pa / 1e5, fd.dP_vana_pa / 1e5),
        ("Boru + fitting sonrası", (fd.P_vana_sonrasi_pa - fd.dP_boru_pa - fd.dP_fitting_pa) / 1e5,
         (fd.dP_boru_pa + fd.dP_fitting_pa) / 1e5),
        ("Manifold (P1)", fd.P1_efektif_pa / 1e5, fd.dP_manifold_pa / 1e5),
    ], columns=["İstasyon", "Basınç [bar]", "Kayıp [bar]"])
    c1, c2 = st.columns([1, 1])
    c1.markdown("**Besleme hattı basınç zinciri**")
    c1.dataframe(zincir.round(3), hide_index=True, **GENIS)
    if man is not None:
        c1.caption(f"Manifold kaybı = port {fd.dP_port_pa/1e5:.3f} bar + dönüş {fd.dP_donus_pa/1e5:.3f} bar")
    c2.markdown("**Hat**")
    c2.write(f"Hat hızı {fd.v_hat_m_s:.2f} m/s · hat/çıkış alan oranı {fd.alan_orani_hat_orifis:.1f} (4'ün üstü önerilir)  \n"
             f"Vana boğulma sınırı ΔP_kr = {fd.dP_vana_bogulma_pa/1e5:.2f} bar, vana ΔP = {fd.dP_vana_pa/1e5:.2f} bar → "
             f"{'**KAVİTASYON**' if fd.vana_kavitasyon else 'OK'}  \n"
             f"Manifold alt-soğutma {fd.alt_sogutma_manifold_bar:.2f} bar · maks. tank sıcaklığı {fd.T_tank_max_K-273.15:.1f} °C")

    if man is None:
        st.info("Ayrıntılı manifold analizi annulus modunda yapılır. Orifis modunda galeri hızı "
                f"{fd.v_manifold_m_s:.2f} m/s.")
    else:
        st.markdown(f"**Manifold** · {g['mf_Di']:.1f}/{g['mf_Do']:.1f} mm × {g['mf_L']:.1f} mm, "
                    f"{int(g['mf_N'])} × Ø{g['mf_d']:.1f} port · {bad} {d_sabit:.3f} mm ile")
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric("Hacim [cm³]", f"{man.V_cm3:.1f}", f"{man.m_n2o_g:.1f} g N2O", delta_color="off")
        a2.metric("Alt-soğutma", f"{man.alt_sogutma_bar:.2f} bar", f"{man.alt_sogutma_K:.1f} K", delta_color="off")
        a3.metric("Dağılım q/ΔP [%]", f"{man.dagilim_pct:.2f}", help="Çevresel dinamik basınç / enjektör ΔP. <%5 önerilir")
        a4.metric("Çevresel / annulus alan", f"{man.oran_cevresel:.1f}", help=">3 önerilir")
        a5.metric("Port σ", f"{man.sigma_port:.1f}", f"σ_i {g['mf_sig']:.1f}", delta_color="off")
        a6.metric("Helmholtz [Hz]", f"{man.f_helmholtz_Hz:.0f}", help="Manifold esnekliği + annulus ataleti, kaba gösterge")

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Hal durumu**")
        c1.dataframe(pd.DataFrame([
            ("Basınç [bar]", f"{man.P_bar:.2f}"), ("Sıcaklık [°C]", f"{man.T_C:.2f}"), ("Faz", man.faz),
            ("Pv(T) [bar]", f"{man.Pv_bar:.2f}"), ("Tsat(P) [°C]", f"{man.Tsat_C:.1f}"),
            ("Kuruluk x [%]", f"{man.x*100:.1f}"), ("Void α [%]", f"{man.alpha*100:.0f}"),
            ("Buhar hacmi [cm³]", f"{man.V_buhar_cm3:.2f}"), ("Yoğunluk [kg/m³]", f"{man.rho_mix:.1f}"),
            ("Kalma süresi [ms]", f"{man.kalma_suresi_ms:.1f}")], columns=["", "Değer"]), hide_index=True, **GENIS)
        c2.markdown("**Kavitasyon öngörüleri**")
        c2.dataframe(pd.DataFrame([
            ("Port jeti σ", f"{man.sigma_port:.2f} → {'KAVİTASYON' if man.port_kavitasyon else 'OK'}"),
            ("Annulus girişi Nurick K / K_kr", f"{man.annulus_K:.3f} / {man.annulus_K_kr:.2f}"),
            ("Annulus girişi", 'KAVİTASYONLU' if man.annulus_kavitasyon else 'kavitasyonsuz'),
            ("Çıkışta flaş (Pv > P2)", 'EVET' if man.annulus_flas else 'hayır'),
            ("Dyer κ", f"{man.kappa_dyer:.2f}"),
            ("τ_r kanal / τ_b kabarcık [µs]", f"{man.tau_r_us:.0f} / {_f(man.tau_b_us, '{:.0f}')}"),
            ("Vana", 'KAVİTASYON' if fd.vana_kavitasyon else 'OK')], columns=["", "Değer"]), hide_index=True, **GENIS)
        c3.markdown("**Akış ve dinamik**")
        c3.dataframe(pd.DataFrame([
            ("Port hızı [m/s]", f"{man.v_port_m_s:.2f}"), ("Çevresel hız [m/s]", f"{man.v_cevresel_m_s:.2f}"),
            ("Eksenel hız [m/s]", f"{man.v_eksenel_m_s:.3f}"), ("Re çevresel", f"{man.Re_cevresel:.2e}"),
            ("Ses hızı [m/s]", f"{man.a_ses_m_s:.0f}"), ("Esneklik [cm³/bar]", f"{man.C_cm3_bar:.4f}"),
            ("Kabarcık yükselme [m/s]", f"{man.u_kabarcik_m_s:.2f}"),
            ("Kabarcık birikmesi", 'RİSK' if man.kabarcik_birikme else 'süpürülür')], columns=["", "Değer"]),
            hide_index=True, **GENIS)
        for w in man.uyarilar:
            st.warning(w)
        st.caption("Annulus girişindeki kavitasyon (Nurick) bu rejimde beklenen durumdur; asıl önemli olan manifoldun kendisinin "
                   "sıvı kalması (alt-soğutma) ve portlarda kavitasyon olmamasıdır. Helmholtz frekansı tek serbestlik "
                   "dereceli kaba bir göstergedir; iki fazda ses hızıyla birlikte hızla düşer.")

# --------------------------------------------------------------------------- sıcaklık
with sekmeler["Sıcaklık"]:
    gecerli = [r for r in tarama if r.mdot]
    st.markdown(f"Tank {op.P_tank_pa/1e5:.1f} bar sabit (He regülatörlü), P2 {op.P2_pa/1e5:.1f} bar, sabit {bad} "
                f"**{d_sabit:.3f} mm**. O/F: yakıt debisi sabit, hedef N2O debisinde O/F = {OF_rpa:.2f}.")
    if gecerli:
        ic = [r.T_C for r in gecerli if not r.durum.startswith('DOYMUŞ') and abs(r.OF_tasarim / OF_rpa - 1) * 100 <= OF_bant]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"O/F ±%{OF_bant:.0f} içinde kalan aralık",
                  f"{min(ic):.1f} – {max(ic):.1f} °C" if ic else "yok", help=f"Tasarım modeli ({MODEL_KISA[op.tasarim_modeli]})")
        k2.metric("Manifoldda kaynama sınırı", f"{fd.T_tank_max_K-273.15:.1f} °C",
                  help="Bu tank sıcaklığının üstünde manifold basıncı Pv'nin altına düşer (nominal debide)")
        k3.metric("dṁ/dT (20 °C civarı)", f"%{uq.dG_dT1_pct_K:+.2f}/K")

        uzun = pd.DataFrame([{"T [°C]": r.T_C, "Model": MODEL_KISA[k], "O/F": m._OF(v, op, OF_rpa)}
                             for r in gecerli for k, v in r.mdot.items() if np.isfinite(v) and k != 'spi'])
        bant_df = pd.DataFrame({"y1": [OF_rpa * (1 - OF_bant / 100)], "y2": [OF_rpa * (1 + OF_bant / 100)]})
        alan = alt.Chart(bant_df).mark_rect(opacity=0.12, color="green").encode(y="y1:Q", y2="y2:Q")
        cz = alt.Chart(uzun).mark_line(point=True).encode(
            x=alt.X("T [°C]:Q"), y=alt.Y("O/F:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N"), strokeWidth=alt.condition(
                alt.datum.Model == MODEL_KISA[op.tasarim_modeli], alt.value(3.5), alt.value(1.3)),
            tooltip=["Model", "T [°C]", alt.Tooltip("O/F:Q", format=".2f")])
        st.altair_chart((alan + cz).properties(height=340, title=f"O/F – tank sıcaklığı (yeşil: ±%{OF_bant:.0f} kabul bandı)"), **GENIS)

        uz2 = pd.DataFrame([{"T [°C]": r.T_C, "Model": MODEL_KISA[k], "gereken": v}
                            for r in gecerli for k, v in r.d_gerekli.items()
                            if k in ('spi',) + m.BANT_MODELLERI and np.isfinite(v) and v < 3 * d_sabit])
        cz2 = alt.Chart(uz2).mark_line(point=True).encode(
            x="T [°C]:Q", y=alt.Y("gereken:Q", title=f"Gereken {bad} [mm]", scale=alt.Scale(zero=False)),
            color="Model:N", tooltip=["Model", "T [°C]", alt.Tooltip("gereken:Q", format=".3f")])
        kural = alt.Chart(pd.DataFrame({"y": [d_sabit]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((cz2 + kural).properties(height=300, title=f"Hedef debi için gereken {bad} (gri: sabit {bad})"), **GENIS)

    tdf = pd.DataFrame([{
        "T [°C]": r.T_C, "Durum": r.durum, "Pv [bar]": r.Pv_bar, "P_man [bar]": r.P_man_bar,
        "Alt-soğ. man. [bar]": r.alt_sogutma_man_bar, "Alt-soğ. [K]": r.alt_sogutma_man_K, "ρ_l": r.rho_l,
        f"ṁ {MODEL_KISA[op.tasarim_modeli]}": r.mdot.get(op.tasarim_modeli, np.nan) if r.mdot else np.nan,
        "Sapma [%]": r.sapma_pct, "O/F tas.": r.OF_tasarim, "O/F min": r.OF_min, "O/F max": r.OF_max,
        f"Gereken {bad} tas. [mm]": r.d_gerekli.get(op.tasarim_modeli, np.nan) if r.d_gerekli else np.nan,
        f"{bad} bant min": r.d_bant_min, f"{bad} bant max": r.d_bant_max, "Nurick K": r.K_nurick,
        "x2 [%]": r.x2 * 100 if np.isfinite(r.x2) else np.nan, "Port σ": r.sigma_port} for r in tarama])
    st.dataframe(tdf.round(3), hide_index=True, **GENIS)
    mdf = pd.DataFrame([{"T [°C]": r.T_C, **{MODEL_KISA[k]: v for k, v in r.mdot.items()}} for r in gecerli])
    with st.expander(f"Sabit {bad}ta model debileri [kg/s]"):
        st.dataframe(mdf.round(3), hide_index=True, **GENIS)
    st.download_button("Sıcaklık tablosunu indir (.csv)", tdf.to_csv(index=False).encode('utf-8-sig'),
                       file_name="n2o_sicaklik_taramasi.csv", mime="text/csv")
    st.caption("Termal bulk ısınma bu taramada ihmal edilir (adyabatik). Tank basıncı sıcaklıktan bağımsız; "
               "Pv(T) tank basıncını geçtiğinde tank içinde, manifold basıncını geçtiğinde hatta kaynama başlar.")

# --------------------------------------------------------------------------- senaryolar
if sen:
    with sekmeler["Senaryolar"]:
        st.markdown(f"Sabit {bad} **{d_sabit:.3f} mm**, hedef {op.m_dot_hedef:.3f} kg/s, O/F kabul ±%{OF_bant:.0f} "
                    f"(ana modele göre karar).")
        ozet = pd.DataFrame([{"Senaryo": s.kod, "Tanım": s.ad, "Giriş": s.giris,
                              "Ana model": MODEL_KISA.get(s.ana_model.split('@')[0], s.ana_model),
                              "ṁ [kg/s]": f"{s.mdot_min:.3f} – {s.mdot_max:.3f}",
                              "O/F": f"{s.OF_min:.2f} – {s.OF_max:.2f}", "Karar": s.kabul} for s in sen])
        st.dataframe(ozet, hide_index=True, **GENIS)

        for s in sen:
            with st.expander(f"{s.kod} · {s.ad}", expanded=(s.kod in ('S2', 'S4'))):
                if s.kod != 'S4':
                    sdf = pd.DataFrame([{"Model": m.SENARYO_MODEL_ADLARI.get(k, k), "ṁ [kg/s]": v,
                                         "Sapma [%]": (v / op.m_dot_hedef - 1) * 100 if np.isfinite(v) else np.nan,
                                         "O/F": s.OF[k], f"Gereken {bad} [mm]": s.d_gerekli[k],
                                         "Ana": "●" if k == s.ana_model else ""} for k, v in s.mdot.items()])
                    st.dataframe(sdf.round(3), hide_index=True, **GENIS)
                else:
                    t4 = pd.DataFrame([{"x1 [%]": r['x1'] * 100, "Model": MODEL_KISA[k], "ṁ": v, "O/F": m._OF(v, op, OF_rpa)}
                                       for r in s.tarama for k, v in r['mdot'].items()])
                    bant_df = pd.DataFrame({"y1": [OF_rpa * (1 - OF_bant / 100)], "y2": [OF_rpa * (1 + OF_bant / 100)]})
                    alan = alt.Chart(bant_df).mark_rect(opacity=0.12, color="green").encode(y="y1:Q", y2="y2:Q")
                    c4 = alt.Chart(t4).mark_line(point=True).encode(
                        x=alt.X("x1 [%]:Q", title="Giriş kuruluğu x1 [%]"), y=alt.Y("O/F:Q", scale=alt.Scale(zero=False)),
                        color="Model:N", tooltip=["Model", "x1 [%]", alt.Tooltip("ṁ:Q", format=".3f"), alt.Tooltip("O/F:Q", format=".2f")])
                    st.altair_chart((alan + c4).properties(height=300), **GENIS)
                    s4df = pd.DataFrame([{"x1 [%]": r['x1'] * 100,
                                          **{f"ṁ {MODEL_KISA[k]}": v for k, v in r['mdot'].items()},
                                          **{f"{bad} {MODEL_KISA[k]}": v for k, v in r['d'].items()},
                                          "Manifold void [%]": r.get('alpha_man', np.nan) * 100,
                                          "Buhar [cm³]": r.get('V_buhar_cm3', np.nan),
                                          "Ses hızı [m/s]": r.get('a_ses', np.nan),
                                          "Helmholtz [Hz]": r.get('f_h', np.nan)} for r in s.tarama])
                    st.dataframe(s4df.round(3), hide_index=True, **GENIS)
                if s.man is not None:
                    st.caption(f"Manifold hali: {s.man.faz} · {s.man.P_bar:.2f} bar · alt-soğutma {s.man.alt_sogutma_bar:.2f} bar · "
                               f"void %{s.man.alpha*100:.0f} · buhar {s.man.V_buhar_cm3:.2f} cm³ · ses hızı {s.man.a_ses_m_s:.0f} m/s · "
                               f"Helmholtz ~{s.man.f_helmholtz_Hz:.0f} Hz")
                for n in s.notlar:
                    st.write("• " + n)
        st.caption("S1/S2: nominal sıvı giriş, hat kaybı bağlaşık. S3/S4: nominal manifold basıncında doymuş karışım "
                   "(T1 = Tsat(P1)), hat kaybı bağlaşımı yok. He basınçlandırmalı sistemde S3/S4 ancak N2O hatta ısınırsa "
                   "veya basınç düşerse oluşur; enjektör girişindeki PT + TC hangi senaryoda olunduğunu gösterir.")

# --------------------------------------------------------------------------- termal
with sekmeler["Termal"]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yanma sonu gaz yüzü [°C]", f"{t.Tw_gaz_tb_K-273.15:.0f}")
    c2.metric("Yanma sonu N2O yüzü [°C]", f"{t.Tw_n2o_tb_K-273.15:.1f}")
    c3.metric("Depolanan ısı [kJ]", f"{t.E_depolanan_nom_J/1e3:.1f} / {t.E_depolanan_max_J/1e3:.1f}",
              help="Nominal / en kötü")
    c4.metric("Hapsolan N2O kaynatma [kJ]", f"{t.E_hapsolan_kaynama_J/1e3:.1f}",
              help=f"{t.m_hapsolan_kg*1e3:.1f} g N2O")
    st.write(f"{t.mat_etiket}, N2O ıslak alan {t.A_ayna_m2*1e4:.1f} cm². "
             f"h_N2O: galeri {t.h_n2o_manifold:.0f}, kanal girişi {t.h_n2o_giris:.0f}, kanal {t.h_n2o_delik:.0f}, "
             f"aynaya indirgenmiş {t.h_n2o_efektif:.0f} W/m²K. Bulk ısınma +{t.dT_bulk_nom_K:.2f} K. "
             f"Galeri T_sat {t.T_sat_manifold_K-273.15:.1f} °C, T_ONB {t.T_onb_K-273.15:.1f} °C.")
    st.write(f"Kapanma sonrası: {th.t_kapanma_sonrasi_s:.0f} s içinde N2O yüzü max {t.Tw_n2o_soak_max_K-273.15:.0f} °C; "
             f"N2O kaynadıktan sonra denge {t.T_denge_kuru_nom_K-273.15:.0f} °C (nominal), "
             f"{t.T_denge_kuru_K-273.15:.0f} °C (en kötü). "
             f"{'Hapsolan N2O tamamen buharlaşıyor.' if t.hapsolan_tamamen_kaynar else ''}")
    gdf = pd.DataFrame(t.grid, columns=["T_gaz [K]", "h_gaz [W/m²K]", "Gaz yüzü [°C]", "N2O yüzü [°C]",
                                        "Q [kW]", "Bulk ΔT [K]"])
    st.caption(f"Yanma sonu ({th.t_yanma_s:.1f} s) 3×3 gaz sınır şartı zarfı")
    st.dataframe(gdf.round(2), hide_index=True, **GENIS)

# --------------------------------------------------------------------------- belirsizlik
with sekmeler["Belirsizlik"]:
    c1, c2 = st.columns(2)
    c1.markdown(f"**Gereken {bad} (P10 / P50 / P90, mm)**  \n"
                f"Yalnız donanım: {uq.d_hw_P10:.3f} / {uq.d_hw_P50:.3f} / {uq.d_hw_P90:.3f}  \n"
                f"Model seçimi dahil: {uq.d_karma_P10:.3f} / {uq.d_karma_P50:.3f} / {uq.d_karma_P90:.3f}")
    c2.markdown(f"**İlk işleme {uq.ilk_delme_mm:.2f} mm ile beklenen debi (kg/s)**  \n"
                f"Yalnız donanım: {uq.mdot_ilk_hw_P10:.3f} / {uq.mdot_ilk_hw_P50:.3f} / {uq.mdot_ilk_hw_P90:.3f}  \n"
                f"Model seçimi dahil: {uq.mdot_ilk_karma_P10:.3f} / {uq.mdot_ilk_karma_P50:.3f} / {uq.mdot_ilk_karma_P90:.3f}  \n"
                f"Hedef {op.m_dot_hedef:.3f}")
    tdf = pd.DataFrame([{"Etken": k, "min": v[0], "max": v[1], "fark": v[2]} for k, v in uq.tornado.items()])
    torn = alt.Chart(tdf).mark_bar(height=16).encode(
        y=alt.Y("Etken:N", sort=list(tdf["Etken"]), title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("min:Q", title=f"Gereken {bad} [mm]", scale=alt.Scale(zero=False)), x2="max:Q",
        tooltip=["Etken", alt.Tooltip("min:Q", format=".3f"), alt.Tooltip("max:Q", format=".3f"),
                 alt.Tooltip("fark:Q", format=".3f")])
    kural = alt.Chart(pd.DataFrame({"x": [b.d_tasarim_mm]})).mark_rule(color="gray").encode(x="x:Q")
    st.altair_chart((torn + kural).properties(height=40 * len(tdf)), **GENIS)
    st.caption(f"Tornado: her etken tek başına değiştirildiğinde gereken {bad}ın aralığı. Gri çizgi: tasarım modeli. "
               f"d(lnG)/d(lnP1) = {uq.dlnG_dlnP1:.2f}, dG/dT1 = %{uq.dG_dT1_pct_K:+.2f}/K.")
    if R['kal_ozet']:
        st.markdown("**Kalibrasyon çarpanları**")
        st.dataframe(pd.DataFrame([{"Model": m.MODEL_ADLARI[k], "k": v[0], "std": v[1], "n": v[2]}
                                   for k, v in R['kal_ozet'].items()]).round(3), hide_index=True)

# --------------------------------------------------------------------------- makale
with sekmeler["Makale kontrolü"]:
    vv = R['vv']
    st.write(f"EUCASS 2023-728 sıvı Test 1: doymuş N2O, T1 = {vv.T1_K:.0f} K (varsayım), P1 = {vv.P1_bar:.2f} bar → "
             f"{vv.P2_bar:.3f} bar, d = 1 mm, Cd = 0.70. Deney: 0.018 kg/s.")
    vrows = []
    for r in vv.satirlar:
        vrows.append({"Model": m.MODEL_ADLARI[r.model], "Kod real_ns [kg/s]": round(r.kod_real_ns, 4),
                      "Kod cp_cv [kg/s]": round(r.kod_cp_cv, 4),
                      "Makale [kg/s]": r.makale_modeli,
                      "Makaleye göre [%]": round((r.kod_real_ns / r.makale_modeli - 1) * 100) if r.makale_modeli else None,
                      "Deneye göre [%]": None if r.model == 'nurick' else round((r.kod_real_ns / r.deney - 1) * 100)})
    st.dataframe(pd.DataFrame(vrows), hide_index=True, **GENIS)
    st.caption("Yalnızca makale rejimini (doymuş besleme, atmosfere boşaltma) temsil eder; 27 bar karşı basınç için "
               "geçerlilik kanıtı değildir. Nurick doymuş beslemede geçersizdir.")

# --------------------------------------------------------------------------- rapor
with sekmeler["Tam rapor"]:
    st.download_button("Raporu indir (.txt)", R['rapor'], file_name="n2o_enjektor_rapor.txt", mime="text/plain")
    st.code(R['rapor'], language=None)
