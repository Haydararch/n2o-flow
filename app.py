# -*- coding: utf-8 -*-
"""
GAZAP N2O Enjektör Akış Hesabı - Streamlit arayüzü
Hesap motoru: n2o_enjektor_v7_2.py (aynı klasörde olmalı, değiştirilmeden kullanılır)
Yerelde çalıştırma:  streamlit run app.py
"""
import contextlib
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import n2o_enjektor_v7_2 as m

st.set_page_config(page_title="GAZAP N2O Enjektör", page_icon="🔥", layout="wide")

# Streamlit sürüm uyumu: yeni sürümler width='stretch', eski sürümler use_container_width=True ister
try:
    _sv = tuple(int(x) for x in st.__version__.split('.')[:2])
except ValueError:
    _sv = (1, 0)
GENIS = {'width': 'stretch'} if _sv >= (1, 46) else {'use_container_width': True}

MALZEMELER = ['Inconel_718', 'AISI_304', 'AISI_316L', 'Al_6061_T6', 'CuCrZr']
MODEL_KISA = {
    'spi': 'SPI', 'spc_liq': 'SPC sıvı', 'hem_lokal': 'HEM yerel', 'nurick': 'Nurick',
    'hem_c': 'HEMc', 'dyer': 'Dyer', 'dyer_comp': 'Dyer-komp.', 'fml': 'FML', 'dual_spc': 'Dual-SPC',
}


# --------------------------------------------------------------------------- hesap
@st.cache_data(show_spinner=False)
def _vv():
    return m.vv_makale_karsilastirmasi()


@st.cache_data(show_spinner=False, max_entries=64)
def hesapla(g: dict, kal_satirlari: tuple):
    op = m.OperatingConfig(
        m_dot_rpa=g['m_dot_rpa'], eta_atomizasyon=g['eta'],
        P_tank_pa=g['P_tank_bar'] * 1e5, T_tank_K=g['T_tank_C'] + 273.15, P2_pa=g['P2_bar'] * 1e5,
        exponent_mode=g['exponent_mode'], vap_ref_mode=g['vap_ref_mode'], slip_mode=g['slip_mode'],
        tasarim_modeli=g['tasarim_modeli'],
    )
    if g['geometri'] == 'orifis':
        orif = m.OrificeConfig(
            geometri='orifis', Cd_tekfaz=g['Cd'], N_orifis=g['N'], L_orifis_mm=g['L'],
            inlet_edge_type=g['edge'], Cc_override=g['Cc_override'], drill_tolerance_mm=g['tol'])
    else:
        orif = m.OrificeConfig(
            geometri='annulus', Cd_tekfaz=g['Cd'], D_ic_mm=g['D_ic'], L_orifis_mm=g['L'],
            inlet_edge_type=g['edge'], Cc_override=g['Cc_override'], drill_tolerance_mm=g['tol'],
            eksantriklik_mm=g['eks'])
    feed = m.FeedSystemConfig(
        D_hat_mm=g['D_hat'], L_hat_m=g['L_hat'], eps_boru_mm=g['eps'], K_tank_cikis=g['K_tank'],
        Cv_ana_vana=g['Cv'], FL_vana=g['FL'], K_fitting_toplam=g['K_fit'], K_manifold_giris=g['K_man'],
        D_manifold_ic_mm=g['D_man'], H_manifold_ic_mm=g['H_man'])
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
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.rapor_yazdir(op, orif, feed, th, vv, sol, tablo, uq, kal_ozet, vv_yazdir=True)
    return dict(op=op, orif=orif, th=th, sol=sol, tablo=tablo, uq=uq, kal_ozet=kal_ozet, vv=vv,
                rapor=buf.getvalue())


# --------------------------------------------------------------------------- girdiler
st.sidebar.title("Girdiler")
geometri = st.sidebar.radio("Çıkış geometrisi", ['orifis', 'annulus'], horizontal=True,
                            format_func=lambda s: 'Orifis (N delik)' if s == 'orifis' else 'Annulus (halka)')

with st.sidebar.form("girdiler"):
    g = {'geometri': geometri}
    with st.expander("İşletme", expanded=True):
        g['m_dot_rpa'] = st.number_input("RPA N2O debisi [kg/s]", 0.01, 20.0, 1.09608, 0.001, format="%.5f")
        g['eta'] = st.number_input("Atomizasyon verimi η [-]", 0.30, 1.00, 0.90, 0.01,
                                   help="Hedef debi = RPA debisi / η")
        g['P_tank_bar'] = st.number_input("Tank basıncı [bar]", 5.0, 72.0, 62.0, 0.5)
        g['T_tank_C'] = st.number_input("Tank sıvı sıcaklığı [°C]", -30.0, 36.0, 19.95, 0.5)
        g['P2_bar'] = st.number_input("Oda basıncı P2 [bar]", 1.0, 70.0, 27.0, 0.5)

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
            g['D_ic'] = st.number_input("Halka iç çapı [mm]", 1.0, 200.0, 22.0, 0.5)
            g['eks'] = st.number_input("Eksantriklik [mm]", 0.0, 1.0, 0.02, 0.005, format="%.3f")
            g['Cd'] = st.number_input("Cd tek faz [-]", 0.31, 1.00, 0.75, 0.01,
                                      help="Kavitasyonsuz tek faz Cd. İnce aralıkta sürtünmeyle ~0.75-0.8.")
            b_min, b_max, b_adim = 0.35, 0.85, 0.05
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

    with st.expander("Besleme hattı"):
        g['D_hat'] = st.number_input("Hat iç çapı [mm]", 2.0, 100.0, 15.7, 0.1)
        g['L_hat'] = st.number_input("Hat uzunluğu [m]", 0.05, 20.0, 1.0, 0.05)
        g['eps'] = st.number_input("Pürüzlülük [mm]", 0.0, 1.0, 0.015, 0.001, format="%.4f")
        g['K_tank'] = st.number_input("K tank çıkışı [-]", 0.0, 10.0, 0.5, 0.1)
        g['Cv'] = st.number_input("Ana vana Cv", 0.1, 100.0, 5.5, 0.1)
        g['FL'] = st.number_input("Vana FL [-]", 0.3, 1.0, 0.60, 0.01)
        g['K_fit'] = st.number_input("K dirsek/rakor toplam [-]", 0.0, 50.0, 0.8, 0.1)
        g['K_man'] = st.number_input("K manifold girişi [-]", 0.0, 20.0, 1.5, 0.1)
        g['D_man'] = st.number_input("Galeri çapı [mm]", 5.0, 300.0, 60.0, 1.0)
        g['H_man'] = st.number_input("Galeri yüksekliği [mm]", 0.5, 100.0, 10.0, 0.5)

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
st.caption("Alaz Space · hesap motoru V7.2 · yalnızca N2O tarafı (yakıt, O/F, yanma yok). "
           "Sonuç tek bir değer değil, modeller arası bant: delikleri bandın küçük ucundan işleyip test sonrası büyütün.")

if gonder or 'sonuc' not in st.session_state:
    try:
        with st.spinner("Hesaplanıyor (ilk çalıştırma ~20-40 s sürebilir)..."):
            st.session_state['sonuc'] = hesapla(g, kal_satirlari)
            st.session_state['hata'] = None
    except ValueError as e:
        st.session_state['hata'] = str(e)

if st.session_state.get('hata'):
    st.error(st.session_state['hata'])
    st.stop()

R = st.session_state['sonuc']
op, orif, th, sol, tablo, uq = R['op'], R['orif'], R['th'], R['sol'], R['tablo'], R['uq']
fl, b, fd, t, si = sol.fl, sol.boy, sol.fd, sol.th, sol.st_in
bad = 'aralık' if orif.annulus else 'çap'
geo = f"Annulus, iç çap {orif.D_ic_mm:.2f} mm" if orif.annulus else f"Orifis, N = {orif.N_orifis}"

# --------------------------------------------------------------------------- özet
st.subheader(f"Özet · {geo} · hedef {op.m_dot_hedef:.4f} kg/s")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"{bad.capitalize()} bandı [mm]", f"{b.bant_min_mm:.2f}–{b.bant_max_mm:.2f}",
          help=f"{MODEL_KISA[b.bant_min_model]} .. {MODEL_KISA[b.bant_max_model]}")
k2.metric("İlk işleme [mm]", f"{b.ilk_delme_mm:.2f}",
          help=(f"Dış çap {orif.D_dis_mm(b.ilk_delme_mm):.3f} mm" if orif.annulus else "Bandın küçük ucu"))
k3.metric(f"Tasarım modeli {bad} [mm]", f"{b.d_tasarim_mm:.3f}", help=MODEL_KISA[op.tasarim_modeli])
k4.metric("Manifold basıncı [bar]", f"{fd.P1_efektif_pa/1e5:.2f}", f"-{fd.dP_toplam_pa/1e5:.2f} hat kaybı",
          delta_color="off")
k5.metric("Maks. tank sıcaklığı [°C]", f"{fd.T_tank_max_K-273.15:.1f}",
          f"şu an {op.T_tank_K-273.15:.1f}", delta_color="off")
k6.metric("Kapanma sonrası ayna [°C]", f"{t.T_denge_kuru_nom_K-273.15:.0f} / {t.T_denge_kuru_K-273.15:.0f}",
          help="Nominal / en kötü zarf")

uyarilar = list(dict.fromkeys(list(si.uyarilar) + fl.uyarilar + b.uyarilar + fd.uyarilar))
durum = t.durum_ozeti
(st.error if durum.startswith("KRİTİK") else st.warning if durum.startswith(("UYARI", "SOAKBACK")) else st.info)(
    "Termal: " + durum)
if uyarilar:
    with st.expander(f"Uyarılar ({len(uyarilar)})", expanded=False):
        for w in uyarilar:
            st.write("• " + w)

sekmeler = st.tabs(["Modeller", f"{bad.capitalize()} tablosu", "Besleme", "Termal",
                    "Belirsizlik", "Makale kontrolü", "Tam rapor"])

# --------------------------------------------------------------------------- modeller
with sekmeler[0]:
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
        f"**Orifis girişi**  \n"
        f"{si.P_pa/1e5:.2f} bar, {si.T_K-273.15:.2f} °C, alt-soğutma {si.alt_sogutma_pa/1e5:.2f} bar  \n"
        f"ρ_l = {si.rho_liq:.1f} kg/m³, Pv(T1) = {si.Pv_pa/1e5:.2f} bar  \n"
        f"**Jet hızı / momentum (TMR için)**  \n"
        f"{fl.v_jet_sivi:.0f} – {fl.v_jet_hem:.0f} m/s → {op.m_dot_hedef*fl.v_jet_sivi:.0f} – "
        f"{op.m_dot_hedef*fl.v_jet_hem:.0f} N")
    st.caption("Referans: banda girmez (SPI/SPC fiziksel üst sınır, HEM yerel aşırı kötümser, orijinal Dyer makalede en kötü). "
               "BANT: bu koşulda savunulabilir beş model. TASARIM: termal, tornado ve tolerans için kullanılan tek boyut.")

# --------------------------------------------------------------------------- boyut tablosu
with sekmeler[1]:
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

# --------------------------------------------------------------------------- besleme
with sekmeler[2]:
    zincir = pd.DataFrame([
        ("Tank", op.P_tank_pa / 1e5, 0.0),
        ("Tank çıkışı", fd.P_vana_oncesi_pa / 1e5, fd.dP_tank_cikis_pa / 1e5),
        ("Vana sonrası", fd.P_vana_sonrasi_pa / 1e5, fd.dP_vana_pa / 1e5),
        ("Boru + fitting sonrası", (fd.P_vana_sonrasi_pa - fd.dP_boru_pa - fd.dP_fitting_pa) / 1e5,
         (fd.dP_boru_pa + fd.dP_fitting_pa) / 1e5),
        ("Manifold (P1)", fd.P1_efektif_pa / 1e5, fd.dP_manifold_pa / 1e5),
    ], columns=["İstasyon", "Basınç [bar]", "Kayıp [bar]"])
    st.dataframe(zincir.round(3), hide_index=True, **GENIS)
    c1, c2, c3 = st.columns(3)
    c1.metric("Hat hızı [m/s]", f"{fd.v_hat_m_s:.2f}")
    c2.metric("Hat / çıkış alan oranı", f"{fd.alan_orani_hat_orifis:.1f}", help="4'ün üstü önerilir")
    c3.metric("Manifold alt-soğutma [bar]", f"{fd.alt_sogutma_manifold_bar:.2f}")
    st.write(f"Vana boğulma sınırı ΔP_kr = {fd.dP_vana_bogulma_pa/1e5:.2f} bar, vana ΔP = {fd.dP_vana_pa/1e5:.2f} bar → "
             f"{'**KAVİTASYON**' if fd.vana_kavitasyon else 'OK'}. Galeri hızı {fd.v_manifold_m_s:.2f} m/s.")

# --------------------------------------------------------------------------- termal
with sekmeler[3]:
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
with sekmeler[4]:
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
with sekmeler[5]:
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
with sekmeler[6]:
    st.download_button("Raporu indir (.txt)", R['rapor'], file_name="n2o_enjektor_rapor.txt", mime="text/plain")
    st.code(R['rapor'], language=None)
