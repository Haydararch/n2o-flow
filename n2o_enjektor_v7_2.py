# -*- coding: utf-8 -*-
"""
===============================================================================
N2O ENJEKTÖR AKIŞ, BESLEME HATTI VE AYNA TERMAL HESABI  (V7.2: orifis + annulus)
===============================================================================
Kapsam
  Yalnızca N2O tarafı. Yakıt, O/F, c*, boğaz hesabı YOKTUR.
  N2O; besleme hattı -> manifold/galeri -> aynadaki eksenel orifisler -> oda.
  Orifis çıkış (oda) basıncı sabit girdidir (P2).

Kaynak
  S. La Luna, D. Zuin, F. Maggi, "N2O Mass Flow Modelling: State-of-the-Art and
  The New Dual SPC Model", EUCASS 2023-728 (DOI 10.13009/EUCASS2023-728).
  FML = Foletti-Magni-La Luna modeli (makalenin 2.5 bölümü).
  Nurick kavitasyon modeli: W.H. Nurick, "Orifice Cavitation and Its Effect on
  Spray Mixing", J. Fluids Eng. 98 (1976).

Önemli uyarı (bu sürümün temel felsefesi)
  Makaledeki deneyler DOYMUŞ sıvı, 1 mm orifis ve ATMOSFERE boşaltma koşulunda
  yapılmıştır. Bu enjektör ise alt-soğutulmuş (basınçlandırılmış) N2O ile ~27 bar
  karşı basınçta çalışır. Makalenin hata istatistikleri (MAE, yanlılık) bu
  koşula TAŞINAMAZ. Bu yüzden:
    * Tek bir "doğru" model yoktur; kod tüm modellerden bir ÇAP/DEBİ BANDI üretir.
    * Kalibrasyon çarpanı yalnızca kendi soğuk akış testlerinizden türetilir
      (bkz. [9] KALİBRASYON). Test yoksa çarpan = 1.0.
    * Delikler bandın KÜÇÜK ucundan delinir, testten sonra raybalanarak büyütülür.

Spyder: hücreler '# %%' ile ayrılmıştır. Gereken paketler:
  pip install CoolProp thermo ht scipy numpy
===============================================================================
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import CoolProp.CoolProp as CP
from thermo import Chemical
from scipy.optimize import root_scalar, minimize_scalar, fsolve
import ht

FLUID = 'NitrousOxide'
P_CRIT = CP.PropsSI('Pcrit', FLUID)
T_CRIT = CP.PropsSI('Tcrit', FLUID)

# Model anahtarları ve rapor adları
MODEL_ADLARI: Dict[str, str] = {
    'spi':       'SPI (Eq.2) - kavitasyonsuz sıvı [referans]',
    'spc_liq':   'SPC sıvı (Eq.9-11) [referans]',
    'hem_lokal': 'HEM yerel (P2, Eq.16) [referans]',
    'nurick':    'Nurick kavitasyonlu SPI',
    'hem_c':     'HEMc (basınç taramalı boğulma)',
    'dyer':      'Dyer (Eq.26: SPI + yerel HEM)',
    'dyer_comp': 'Dyer-komp. (Eq.28: SPC + HEMc)',
    'fml':       'FML sıvı (Eq.31)',
    'dual_spc':  'Dual-SPC (Bölüm 3)',
}
# Tasarım bandını oluşturan modeller (referanslar ve makalede en kötü çıkan
# orijinal Dyer banda dahil edilmez; raporda yine gösterilir).
BANT_MODELLERI: Tuple[str, ...] = ('nurick', 'hem_c', 'dyer_comp', 'fml', 'dual_spc')

# Nurick daralma katsayısı Cc (yaklaşık literatür değerleri; su testiyle güncelleyin)
CC_KENAR: Dict[str, float] = {'sharp': 0.61, 'chamfered': 0.68, 'rounded': 0.80}
# Belirsizlik analizinde Cd_tekfaz'ın etrafında taranan yarı-genişlik (±)
CD_BELIRSIZLIK: float = 0.08


# %% [1] GİRDİLER
@dataclass(frozen=True)
class OperatingConfig:
    m_dot_rpa: float = 1.09608          # RPA'dan gelen N2O debisi [kg/s]
    eta_atomizasyon: float = 0.90       # Varsayılan atomizasyon verimi [-] -> hedef debi = RPA / eta
    P_tank_pa: float = 62.0e5           # Tank basıncı [Pa]
    T_tank_K: float = 293.10            # Tanktaki sıvı N2O sıcaklığı [K]
    P2_pa: float = 27.0e5               # Orifis çıkış (oda) basıncı [Pa]
    manuel_x1: Optional[float] = None   # Opsiyonel: giriş kuruluk derecesi (iki fazlı besleme) [-]
    exponent_mode: str = 'real_ns'      # 'real_ns' (Eq.6, rho*c^2/P) veya 'cp_cv'
    vap_ref_mode: str = 'T1_sat'        # 'T1_sat' (Pv(T1), fiziksel) veya 'P1_sat'
    slip_mode: str = 'zivi'             # 'zivi' veya 'homogeneous' (S = 1)
    tasarim_modeli: str = 'dual_spc'    # Termal ve tornado için referans model (BANT_MODELLERI'nden)

    def __post_init__(self):
        if not (0.0 < self.eta_atomizasyon <= 1.0):
            raise ValueError("eta_atomizasyon (0, 1] aralığında olmalı.")
        if self.P2_pa >= self.P_tank_pa:
            raise ValueError("P2, tank basıncından küçük olmalı.")
        if self.exponent_mode not in ('real_ns', 'cp_cv'):
            raise ValueError("exponent_mode: 'real_ns' veya 'cp_cv'.")
        if self.vap_ref_mode not in ('T1_sat', 'P1_sat'):
            raise ValueError("vap_ref_mode: 'T1_sat' veya 'P1_sat'.")
        if self.slip_mode not in ('zivi', 'homogeneous'):
            raise ValueError("slip_mode: 'zivi' veya 'homogeneous'.")
        if self.tasarim_modeli not in MODEL_ADLARI:
            raise ValueError(f"tasarim_modeli şunlardan biri olmalı: {list(MODEL_ADLARI)}")

    @property
    def m_dot_hedef(self) -> float:
        return self.m_dot_rpa / self.eta_atomizasyon


@dataclass(frozen=True)
class OrificeConfig:
    """
    Çıkış geometrisi. 'orifis': N adet özdeş silindirik delik, boyut = delik çapı.
    'annulus': tek halka aralık (iç çap D_ic_mm sabit), boyut = radyal aralık (gap).
    Akı modelleri (G) geometriden bağımsızdır; geometri yalnızca alan <-> boyut dönüşümünü,
    hidrolik çapı, ıslak duvar alanını ve toleransın neye uygulandığını belirler.
    """
    geometri: str = 'orifis'            # 'orifis' veya 'annulus'
    Cd_tekfaz: float = 0.70             # Kavitasyonsuz tek faz deşarj katsayısı [-] (su testiyle ölçün)
    N_orifis: int = 8                   # yalnızca 'orifis'
    D_ic_mm: Optional[float] = None     # yalnızca 'annulus': halkanın iç çapı [mm]
    eksantriklik_mm: float = 0.02       # yalnızca 'annulus': iç/dış merkez kaçıklığı [mm]
    L_orifis_mm: float = 8.0            # Kanal boyu (orifis boyu / annulus boyu) [mm]
    inlet_edge_type: str = 'sharp'      # 'sharp', 'chamfered', 'rounded' -> Nurick Cc
    Cc_override: Optional[float] = None # Ölçülmüş Cc varsa buraya
    drill_tolerance_mm: float = 0.02    # İşleme toleransı [±mm] (orifis: çap, annulus: aralık)

    def __post_init__(self):
        if self.geometri not in ('orifis', 'annulus'):
            raise ValueError("geometri: 'orifis' veya 'annulus'.")
        if self.geometri == 'orifis' and self.N_orifis < 1:
            raise ValueError("N_orifis >= 1 olmalı.")
        if self.geometri == 'annulus' and (self.D_ic_mm is None or self.D_ic_mm <= 0):
            raise ValueError("annulus için D_ic_mm (iç çap) girilmeli.")
        if self.inlet_edge_type not in CC_KENAR:
            raise ValueError(f"inlet_edge_type: {list(CC_KENAR)}")
        if not (0.3 < self.Cd_tekfaz <= 1.0):
            raise ValueError(
                f"Cd_tekfaz = {self.Cd_tekfaz} makul değil (0.3-1.0). Cd_tekfaz KAVİTASYONSUZ tek faz katsayısıdır; "
                f"kavitasyon (Nurick) ve flaş (HEM/Dual-SPC vb.) modellerde ayrıca hesaplanır, onlar için Cd düşürülmez. "
                f"Tipik: keskin girişli kısa delik/kanal (L/Dh 2-8) 0.75-0.82, ince levha orifis 0.61, yuvarlatılmış giriş 0.85-0.95.")

    @property
    def Cc(self) -> float:
        return self.Cc_override if self.Cc_override is not None else CC_KENAR[self.inlet_edge_type]

    @property
    def annulus(self) -> bool:
        return self.geometri == 'annulus'

    @property
    def boyut_adi(self) -> str:
        return 'aralık' if self.annulus else 'çap'

    def alan_m2(self, boyut_mm):
        """Boyut [mm] (çap veya aralık) -> toplam akış alanı [m2]. NumPy dizisiyle de çalışır."""
        b = np.asarray(boyut_mm) * 1e-3
        if self.annulus:
            Di = self.D_ic_mm * 1e-3
            return np.pi / 4.0 * ((Di + 2.0 * b) ** 2 - Di ** 2)
        return self.N_orifis * np.pi * b ** 2 / 4.0

    def boyut_mm(self, A_m2):
        """Toplam akış alanı [m2] -> boyut [mm]. NumPy dizisiyle de çalışır."""
        A = np.asarray(A_m2)
        if self.annulus:
            Di = self.D_ic_mm * 1e-3
            return (np.sqrt(Di ** 2 + 4.0 * A / np.pi) - Di) / 2.0 * 1e3
        return np.sqrt(4.0 * A / (self.N_orifis * np.pi)) * 1e3

    def hidrolik_cap_m(self, boyut_mm: float) -> float:
        return 2.0 * boyut_mm * 1e-3 if self.annulus else boyut_mm * 1e-3

    def islak_cevre_m(self, boyut_mm: float) -> float:
        """Akış kanalının toplam ıslak çevresi [m] (duvar alanı = çevre x boy)."""
        if self.annulus:
            Di = self.D_ic_mm * 1e-3
            return np.pi * (2.0 * Di + 2.0 * boyut_mm * 1e-3)
        return self.N_orifis * np.pi * boyut_mm * 1e-3

    def D_dis_mm(self, boyut_mm: float) -> float:
        return self.D_ic_mm + 2.0 * boyut_mm if self.annulus else float('nan')


@dataclass(frozen=True)
class FeedSystemConfig:
    D_hat_mm: float = 15.7              # Hat iç çapı [mm] (1/2" Sch40)
    L_hat_m: float = 0.80               # Hat uzunluğu [m]
    eps_boru_mm: float = 0.015          # Pürüzlülük [mm]
    K_tank_cikis: float = 0.5           # Tank çıkışı / sifon girişi kayıp katsayısı [-]
    Cv_ana_vana: float = 5.5            # Ana vana Cv [gal/min @ 1 psi]
    FL_vana: float = 0.60               # Vana basınç geri kazanım faktörü (küresel vana ~0.6) [-]
    K_fitting_toplam: float = 0.8       # Dirsek/rakor toplam K [-]
    K_manifold_giris: float = 1.5       # Manifold girişi (genişleme + dönüş) K [-]
    D_manifold_ic_mm: float = 60.0      # Galeri ortalama çapı [mm]
    H_manifold_ic_mm: float = 10.0      # Galeri yüksekliği (radyal akış aralığı) [mm]


@dataclass(frozen=True)
class ThermalConfig:
    # Oda tarafı sınır şartı zarfı (RPA / Bartz çıktılarından girilmeli)
    T_gaz_eff_low_K: float = 1400.0
    T_gaz_eff_nom_K: float = 2050.0
    T_gaz_eff_high_K: float = 2453.0    # RPA Taw (GAZAP oda, 2180 °C)
    h_gaz_low: float = 500.0
    h_gaz_nominal: float = 950.0
    h_gaz_high: float = 1600.0
    # Ayna geometrisi
    D_ayna_mm: float = 83.4             # N2O aynası dış çapı (GAZAP liner iç çapı) [mm]  <- kontrol edin
    D_merkez_mm: float = 20.0           # Ortadaki yakıt gövdesinin (yıldız başlık) çapı [mm] <- VARSAYIM, girin
    malzeme: str = 'AISI_304'           # 'AISI_304','AISI_316L','Inconel_718','Al_6061_T6','CuCrZr'
    t_yanma_s: float = 5.0
    # Kapanma sonrası (soakback)
    t_kapanma_sonrasi_s: float = 30.0   # Kapanmadan sonra izlenecek süre [s]
    h_n2o_durgun: float = 300.0         # Durgun N2O doğal taşınım katsayısı [W/m2K] (VARSAYIM)
    V_hapsolan_cm3: float = 25.0        # Vana ile ayna arasında kalan N2O hacmi [cm3] <- VARSAYIM, girin
    # Yanma sırasındaki ısının bulk akışa karışma oranı (hassasiyet parametresi)
    f_mix_nominal: float = 0.35
    # Duvar sıcaklığı emniyet eşikleri (kullanıcı tanımlı)
    T_wall_caution_K: float = 473.15    # 200 °C
    T_wall_hazard_K: float = 573.15     # 300 °C

    def __post_init__(self):
        if self.D_merkez_mm >= self.D_ayna_mm:
            raise ValueError("D_merkez_mm, D_ayna_mm'den küçük olmalı.")


@dataclass(frozen=True)
class KalibrasyonTesti:
    """Kendi soğuk akış testiniz: ölçülen debi ve test koşulları."""
    P1_bar: float           # Orifis öncesi (manifold) basıncı
    T1_K: float             # Orifis öncesi sıvı sıcaklığı
    P2_bar: float           # Orifis sonrası basınç (atmosfer ya da karşı basınç tankı)
    d_mm: float             # orifis çapı; annulus testinde aralık (gap)
    N_orifis: int           # orifis sayısı (annulus testinde 1 yazın)
    mdot_olculen: float     # [kg/s]
    D_ic_mm: Optional[float] = None   # annulus testi ise halka iç çapı; orifis testinde None


# %% [2] MALZEME KÜTÜPHANESİ
def malzeme_ozellikleri_al(malzeme_adi: str, T_K: float) -> Tuple[float, float, float, str]:
    """(k [W/mK], rho [kg/m3], cp [J/kgK], etiket). Sıcaklığa bağlı doğrusal yaklaşımlar."""
    T_C = float(np.clip(T_K - 273.15, -50.0, 1200.0))
    tablo = {
        'AISI_304':    (14.6 + 0.0127 * T_C, 7930.0, 470.0 + 0.18 * T_C, "AISI 304 Paslanmaz Çelik"),
        'AISI_316L':   (13.4 + 0.0136 * T_C, 7990.0, 465.0 + 0.19 * T_C, "AISI 316L Paslanmaz Çelik"),
        'Inconel_718': (11.4 + 0.0148 * T_C, 8190.0, 435.0 + 0.22 * T_C, "Inconel 718"),
        'Al_6061_T6':  (167.0 - 0.045 * T_C, 2700.0, 896.0 + 0.42 * T_C, "Alüminyum 6061-T6"),
        'CuCrZr':      (325.0 - 0.065 * T_C, 8900.0, 385.0 + 0.09 * T_C, "CuCrZr"),
    }
    if malzeme_adi not in tablo:
        raise ValueError(f"Bilinmeyen malzeme: {malzeme_adi}")
    k, rho, cp, etiket = tablo[malzeme_adi]
    return float(k), float(rho), float(cp), etiket


# %% [3] TERMODİNAMİK DURUM
@lru_cache(maxsize=1024)
def _n2o_transport(T_round_K: float, P_round_pa: float) -> Tuple[float, float, float, float]:
    """thermo ile N2O sıvı mu, k, Pr, sigma (CoolProp'ta N2O taşınım özellikleri yok)."""
    chem = Chemical('nitrous oxide', T=T_round_K, P=P_round_pa)
    if chem.mu is None or chem.k is None:
        raise RuntimeError("thermo N2O taşınım özelliklerini hesaplayamadı.")
    Pr = float(chem.Pr) if chem.Pr is not None else float(chem.mu * chem.Cp / chem.k)
    sigma = float(chem.sigma) if chem.sigma is not None else 1.5e-3
    return float(chem.mu), float(chem.k), Pr, sigma


@dataclass(frozen=True)
class VaporRef:
    label: str
    P_in_pa: float
    T_sat_K: float
    rho: float
    s: float
    h: float
    gamma_cp_cv: float
    ns_real: float
    n_active: float


def _buhar_referansi(P_sat_pa: float, exponent_mode: str, label: str) -> VaporRef:
    T_sat = CP.PropsSI('T', 'P', P_sat_pa, 'Q', 1, FLUID)
    rho_v = CP.PropsSI('D', 'P', P_sat_pa, 'Q', 1, FLUID)
    s_v = CP.PropsSI('S', 'P', P_sat_pa, 'Q', 1, FLUID)
    h_v = CP.PropsSI('H', 'P', P_sat_pa, 'Q', 1, FLUID)
    T_eval = T_sat + 1e-3                     # tek fazlı bölgede türev için mikro ofset
    cp_v = CP.PropsSI('C', 'P', P_sat_pa, 'T', T_eval, FLUID)
    cv_v = CP.PropsSI('O', 'P', P_sat_pa, 'T', T_eval, FLUID)
    ns_v = CP.PropsSI('isentropic_expansion_coefficient', 'P', P_sat_pa, 'T', T_eval, FLUID)
    gam = cp_v / cv_v
    return VaporRef(label, float(P_sat_pa), float(T_sat), float(rho_v), float(s_v), float(h_v),
                    float(gam), float(ns_v), float(ns_v if exponent_mode == 'real_ns' else gam))


@dataclass(frozen=True)
class FluidState:
    P_pa: float
    T_K: float
    Pv_pa: float                # Pv(T1)
    Tsat_P_K: float             # Tsat(P1)
    faz: str
    x: float
    alpha: float                # giriş void oranı (kanal B alan payı)
    rho_mix: float
    rho_liq: float
    s_mix: float
    h_mix: float
    s_liq: float
    gamma_liq: float
    ns_liq: float
    n_active_liq: float
    vap_P1: VaporRef
    vap_T1: VaporRef
    vap_active: VaporRef
    mu_liq: float
    k_liq: float
    Pr_liq: float
    alt_sogutma_pa: float       # P1 - Pv(T1)
    uyarilar: Tuple[str, ...]


def termodinamik_durum_coz(P_pa: float, T_K: float, manuel_x: Optional[float] = None,
                           exponent_mode: str = 'real_ns', vap_ref_mode: str = 'T1_sat',
                           slip_mode: str = 'zivi') -> FluidState:
    """
    Orifis öncesi N2O durumu. İçinde sıvı olan bir hatta P < Pv(T) fiziksel olarak
    mümkün değildir: bu durumda hesap DURDURULUR (eski sürüm sessizce 'kızgın buhar'
    dalına geçip gaz için boyutlandırıyordu).
    """
    if P_pa >= P_CRIT - 1e3:
        raise ValueError(f"P = {P_pa/1e5:.2f} bar, kritik basınca ({P_CRIT/1e5:.2f} bar) eşit/üstünde: model geçersiz.")
    if T_K >= T_CRIT - 0.05:
        raise ValueError(f"T = {T_K:.2f} K, kritik sıcaklığa ({T_CRIT:.2f} K) eşit/üstünde: model geçersiz.")

    uyarilar: List[str] = []
    if P_pa > 0.80 * P_CRIT or T_K > 0.95 * T_CRIT:
        uyarilar.append(f"KRİTİK BÖLGEYE YAKIN: P = %{100*P_pa/P_CRIT:.0f} Pkr, T = %{100*T_K/T_CRIT:.1f} Tkr. "
                        f"Özellikler hızlı değişir, model belirsizliği artar.")

    Pv = CP.PropsSI('P', 'T', T_K, 'Q', 0, FLUID)
    Tsat_P = CP.PropsSI('T', 'P', P_pa, 'Q', 0, FLUID)
    vap_P1 = _buhar_referansi(P_pa, exponent_mode, f"P1_sat ({P_pa/1e5:.1f} bar)")
    vap_T1 = _buhar_referansi(Pv, exponent_mode, f"Pv(T1) ({Pv/1e5:.1f} bar)")

    if manuel_x is not None:
        x_val = float(np.clip(manuel_x, 0.0, 1.0))
        faz = "İKİ FAZLI DOYMUŞ KARIŞIM (MANUEL x)"
        if abs(P_pa - Pv) > 0.5e5:
            uyarilar.append("Manuel x verildi: durum P1'de doymuş kabul edildi, T1 yok sayıldı.")
        rho_l = CP.PropsSI('D', 'P', P_pa, 'Q', 0, FLUID)
        s_l = CP.PropsSI('S', 'P', P_pa, 'Q', 0, FLUID)
        h_l = CP.PropsSI('H', 'P', P_pa, 'Q', 0, FLUID)
        S1 = (rho_l / vap_P1.rho) ** (1.0 / 3.0) if slip_mode == 'zivi' else 1.0
        alpha_val = 0.0 if x_val <= 0 else 1.0 / (1.0 + ((1.0 - x_val) / x_val) * S1 * (vap_P1.rho / rho_l))
        rho_m = 1.0 / (x_val / vap_P1.rho + (1.0 - x_val) / rho_l)
        s_m = x_val * vap_P1.s + (1.0 - x_val) * s_l
        h_m = x_val * vap_P1.h + (1.0 - x_val) * h_l
        T_liq_eval = Tsat_P - 1e-3
        vap_act = vap_P1          # doymuş karışımda P1 = Pv, iki referans aynı
        Pv_eff = P_pa
    elif P_pa > Pv + 100.0:
        faz = "ALT-SOĞUTULMUŞ SIVI"
        x_val, alpha_val = 0.0, 0.0
        rho_l = CP.PropsSI('D', 'P', P_pa, 'T', T_K, FLUID)
        s_l = CP.PropsSI('S', 'P', P_pa, 'T', T_K, FLUID)
        h_l = CP.PropsSI('H', 'P', P_pa, 'T', T_K, FLUID)
        rho_m, s_m, h_m = rho_l, s_l, h_l
        T_liq_eval = T_K
        vap_act = vap_T1 if vap_ref_mode == 'T1_sat' else vap_P1
        Pv_eff = Pv
    elif P_pa >= Pv - 100.0:
        faz = "DOYMUŞ SIVI"
        x_val, alpha_val = 0.0, 0.0
        rho_l = CP.PropsSI('D', 'P', P_pa, 'Q', 0, FLUID)
        s_l = CP.PropsSI('S', 'P', P_pa, 'Q', 0, FLUID)
        h_l = CP.PropsSI('H', 'P', P_pa, 'Q', 0, FLUID)
        rho_m, s_m, h_m = rho_l, s_l, h_l
        T_liq_eval = Tsat_P - 1e-3
        vap_act = vap_P1
        Pv_eff = Pv
    else:
        raise ValueError(
            f"FAZ HATASI: P1 = {P_pa/1e5:.2f} bar < Pv({T_K-273.15:.1f} °C) = {Pv/1e5:.2f} bar. "
            f"Sıvı N2O bu basınçta kaynar (hatta buhar kilidi). Tank basıncını artırın veya "
            f"N2O sıcaklığını en fazla {Tsat_P-273.15:.1f} °C'ye düşürün."
        )

    cp_l = CP.PropsSI('C', 'P', P_pa, 'T', T_liq_eval, FLUID)
    cv_l = CP.PropsSI('O', 'P', P_pa, 'T', T_liq_eval, FLUID)
    ns_l = CP.PropsSI('isentropic_expansion_coefficient', 'P', P_pa, 'T', T_liq_eval, FLUID)
    gam_l = cp_l / cv_l
    mu_l, k_l, Pr_l, _ = _n2o_transport(round(T_liq_eval, 2), round(P_pa, -3))

    if exponent_mode == 'real_ns' and vap_act.ns_real <= 1.005:
        uyarilar.append(f"Buhar izentropik üssü ns = {vap_act.ns_real:.4f} <= 1.005: izotermal limit kullanılıyor.")

    return FluidState(
        P_pa=float(P_pa), T_K=float(T_K), Pv_pa=float(Pv_eff), Tsat_P_K=float(Tsat_P), faz=faz,
        x=x_val, alpha=float(alpha_val), rho_mix=float(rho_m), rho_liq=float(rho_l),
        s_mix=float(s_m), h_mix=float(h_m), s_liq=float(s_l),
        gamma_liq=float(gam_l), ns_liq=float(ns_l),
        n_active_liq=float(ns_l if exponent_mode == 'real_ns' else gam_l),
        vap_P1=vap_P1, vap_T1=vap_T1, vap_active=vap_act,
        mu_liq=mu_l, k_liq=k_l, Pr_liq=Pr_l,
        alt_sogutma_pa=float(P_pa - Pv_eff), uyarilar=tuple(uyarilar)
    )


def durum_op(P_pa: float, T_K: float, op: OperatingConfig) -> FluidState:
    return termodinamik_durum_coz(P_pa, T_K, op.manuel_x1, op.exponent_mode, op.vap_ref_mode, op.slip_mode)


# %% [4] KÜTLESEL AKI MODELLERİ
def _spc_ideal_flux(rho_in: float, P_in: float, P_out: float, n_exp: float) -> Tuple[float, float, bool]:
    """SPC ideal akı (Cd = 1), Eq. 9-11. n -> 1 için analitik izotermal limit. (G, r_kr, boğulma)"""
    if P_out >= P_in:
        return 0.0, 0.5, False
    r = float(np.clip(P_out / P_in, 1e-6, 0.999999))
    if n_exp <= 1.0 + 1e-3:
        r_kr = float(np.exp(-0.5))
        if r <= r_kr:
            return float(np.sqrt(rho_in * P_in * np.exp(-1.0))), r_kr, True
        return float(np.sqrt(2.0 * rho_in * P_in * max(-(r ** 2) * np.log(r), 0.0))), r_kr, False
    r_kr = float((2.0 / (n_exp + 1.0)) ** (n_exp / (n_exp - 1.0)))
    if r <= r_kr:
        G = np.sqrt(n_exp * rho_in * P_in * (2.0 / (n_exp + 1.0)) ** ((n_exp + 1.0) / (n_exp - 1.0)))
        return float(G), r_kr, True
    term = r ** (2.0 / n_exp) - r ** ((n_exp + 1.0) / n_exp)
    G = np.sqrt(2.0 * rho_in * P_in * (n_exp / (n_exp - 1.0)) * max(term, 0.0))
    return float(G), r_kr, False


def _hem_izentropik(s_in: float, h_in: float, P_out_pa: float) -> Tuple[float, float, float]:
    """İzentropik denge genleşmesi -> (G_ideal, rho_out, h_out). Eq. 14-16."""
    s_l = CP.PropsSI('S', 'P', P_out_pa, 'Q', 0, FLUID)
    s_v = CP.PropsSI('S', 'P', P_out_pa, 'Q', 1, FLUID)
    if s_in <= s_l or s_in >= s_v:
        rho_out = CP.PropsSI('D', 'P', P_out_pa, 'S', s_in, FLUID)
        h_out = CP.PropsSI('H', 'P', P_out_pa, 'S', s_in, FLUID)
    else:
        x_out = (s_in - s_l) / (s_v - s_l)
        rho_l = CP.PropsSI('D', 'P', P_out_pa, 'Q', 0, FLUID)
        rho_v = CP.PropsSI('D', 'P', P_out_pa, 'Q', 1, FLUID)
        h_l = CP.PropsSI('H', 'P', P_out_pa, 'Q', 0, FLUID)
        h_v = CP.PropsSI('H', 'P', P_out_pa, 'Q', 1, FLUID)
        rho_out = 1.0 / (x_out / rho_v + (1.0 - x_out) / rho_l)
        h_out = x_out * h_v + (1.0 - x_out) * h_l
    dh = h_in - h_out
    G = rho_out * np.sqrt(2.0 * dh) if dh > 0 else 0.0
    return float(G), float(rho_out), float(h_out)


@lru_cache(maxsize=512)
def _hem_bogulma_taramasi(s_in_round: float, h_in_round: float, P1_round_pa: float) -> Tuple[float, float]:
    """HEM maksimum akı ve boğulma basıncı (P2'den bağımsız; giriş s, h ile önbellekli)."""
    p_grid = np.linspace(1.01e5, P1_round_pa - 1e4, 41)
    g_grid = [_hem_izentropik(s_in_round, h_in_round, p)[0] for p in p_grid]
    i = int(np.argmax(g_grid))
    lo, hi = p_grid[max(0, i - 1)], p_grid[min(len(p_grid) - 1, i + 1)]
    res = minimize_scalar(lambda p: -_hem_izentropik(s_in_round, h_in_round, p)[0],
                          bounds=(lo, hi), method='bounded', options={'xatol': 100.0})
    return float(res.x), float(-res.fun)


@dataclass
class AkisSonucu:
    Cd_tekfaz: float
    Cc: float
    G_ideal: Dict[str, float]           # Cd = 1 (Nurick hariç)
    G_eff: Dict[str, float]             # Cd uygulanmış
    G_kal: Dict[str, float]             # Cd + kullanıcı kalibrasyonu uygulanmış
    # Nurick
    K_nurick: float
    K_kritik: float
    kavitasyonlu: bool
    Cd_nurick: float
    # HEM
    P_hem_bogulma_pa: float
    hem_bogulmus: bool
    # Çıkış durumu
    T2_sat_K: float
    rho_2_liq: float
    rho_2_vap: float
    x_2: float
    alpha_2: float
    alpha_2_S1: float
    S_2_zivi: float
    S_2_aktif: float
    rho_2_homojen: float
    # SPC rejimleri
    r_kr_liq: float
    r_kr_vap: float
    sivi_bogulmus: bool
    buhar_bogulmus: bool
    # Hızlar / tutarlılık
    v_jet_sivi: float                   # sqrt(2 dP / rho_l): flaşsız sıvı jet (alt sınır)
    v_jet_hem: float                    # izentropik denge genleşmesi (tam flaş, üst sınır)
    c_2_wallis: float
    Mach_dual_homojen: float
    kappa_dyer: float
    tau_r_s: float
    uyarilar: List[str]


def iki_fazli_akisi_coz(st: FluidState, P2_pa: float, Cd: float, Cc: float, L_orifis_m: float,
                        slip_mode: str = 'zivi', kalibrasyon: Optional[Dict[str, float]] = None) -> AkisSonucu:
    if P2_pa >= st.P_pa:
        raise ValueError("P2 >= P1: akış yok.")
    if P2_pa >= P_CRIT:
        raise ValueError("P2 kritik basıncın üstünde.")
    uyarilar: List[str] = []
    P1 = st.P_pa
    dP = P1 - P2_pa

    # Çıkış doyma özellikleri
    T2_sat = CP.PropsSI('T', 'P', P2_pa, 'Q', 0, FLUID)
    s2l = CP.PropsSI('S', 'P', P2_pa, 'Q', 0, FLUID)
    s2v = CP.PropsSI('S', 'P', P2_pa, 'Q', 1, FLUID)
    rho2l = CP.PropsSI('D', 'P', P2_pa, 'Q', 0, FLUID)
    rho2v = CP.PropsSI('D', 'P', P2_pa, 'Q', 1, FLUID)
    c2l = CP.PropsSI('A', 'P', P2_pa, 'T', T2_sat - 1e-3, FLUID)
    c2v = CP.PropsSI('A', 'P', P2_pa, 'T', T2_sat + 1e-3, FLUID)

    S_zivi = float((rho2l / rho2v) ** (1.0 / 3.0))     # Eq. 19 / 29
    S_act = 1.0 if slip_mode == 'homogeneous' else S_zivi

    def _x_alpha(s_in: float, S: float) -> Tuple[float, float]:
        x = (s_in - s2l) / (s2v - s2l)                   # Eq. 30
        if x <= 0.0:
            return 0.0, 0.0
        if x >= 1.0:
            return 1.0, 1.0
        return float(x), float(1.0 / (1.0 + ((1.0 - x) / x) * S * (rho2v / rho2l)))   # Eq. 18 / 33

    x2A, aA = _x_alpha(st.s_liq, S_act)
    _, aA_S1 = _x_alpha(st.s_liq, 1.0)
    x2B, aB = _x_alpha(st.vap_active.s, S_act)
    _, aB_S1 = _x_alpha(st.vap_active.s, 1.0)
    a1 = st.alpha
    x2 = (1.0 - a1) * x2A + a1 * x2B
    alpha2 = (1.0 - a1) * aA + a1 * aB
    alpha2_S1 = (1.0 - a1) * aA_S1 + a1 * aB_S1
    rho2_hom = 1.0 / (x2 / rho2v + (1.0 - x2) / rho2l)

    # SPC
    G_spcl, r_kr_l, liq_chk = _spc_ideal_flux(st.rho_liq, P1, P2_pa, st.n_active_liq)
    va = st.vap_active
    G_spcv, r_kr_v, vap_chk = _spc_ideal_flux(va.rho, va.P_in_pa, P2_pa, va.n_active)

    # Dual-SPC (Bölüm 3): kanal alan payları A_A = 1-a1, A_B = a1
    G_dual = ((1.0 - a1) * ((1.0 - aA) * G_spcl + aA * G_spcv)
              + a1 * ((1.0 - aB) * G_spcl + aB * G_spcv))

    # SPI
    G_spi = float(np.sqrt(2.0 * st.rho_liq * dP))

    # HEM
    G_hem_lok, _, h_out_P2 = _hem_izentropik(st.s_mix, st.h_mix, P2_pa)
    P_ch, G_hem_max = _hem_bogulma_taramasi(round(st.s_mix, 3), round(st.h_mix, 2), round(P1, -3))
    if P_ch > P2_pa * 1.001 and G_hem_max > G_hem_lok:
        hem_chk, G_hemc = True, G_hem_max
    else:
        hem_chk, G_hemc, P_ch = False, G_hem_lok, P2_pa

    # Dyer (Eq. 25-26) ve sıkıştırılabilir Dyer (Eq. 28)
    if st.Pv_pa > P2_pa:
        kappa = float(np.sqrt(dP / (st.Pv_pa - P2_pa)))
    else:
        kappa = float('inf')            # Pv <= P2: flaş yok, Dyer -> SPI
    w_hem = 0.0 if np.isinf(kappa) else 1.0 / (1.0 + kappa)
    G_dyer = (1.0 - w_hem) * G_spi + w_hem * G_hem_lok
    G_dyer_comp = (1.0 - w_hem) * G_spcl + w_hem * G_hemc

    # FML sıvı (Eq. 31)
    G_fml = (1.0 - alpha2) * G_spcl + alpha2 * G_hemc

    # Nurick: Cd_kav = Cc*sqrt(K), K = (P1 - Pv)/(P1 - P2). Kavitasyon K < K_kr = (Cd/Cc)^2
    K = (P1 - st.Pv_pa) / dP
    K_kr = (Cd / Cc) ** 2
    Cd_nur = min(Cd, Cc * np.sqrt(max(K, 0.0)))
    kav = Cc * np.sqrt(max(K, 0.0)) < Cd
    if K < 0.05:
        uyarilar.append(f"Nurick: K = {K:.3f} ~ 0 (doymuş besleme). Model bu rejimde geçerli değil.")

    G_ideal = {'spi': G_spi, 'spc_liq': G_spcl, 'hem_lokal': G_hem_lok, 'nurick': G_spi,
               'hem_c': G_hemc, 'dyer': G_dyer, 'dyer_comp': G_dyer_comp, 'fml': G_fml, 'dual_spc': G_dual}
    G_eff = {k: Cd * v for k, v in G_ideal.items()}
    G_eff['nurick'] = float(Cd_nur * G_spi)
    kal = kalibrasyon or {}
    G_kal = {k: v * kal.get(k, 1.0) for k, v in G_eff.items()}

    # Hızlar: momentum akısı (TMR) için bant
    v_sivi = float(np.sqrt(2.0 * dP / st.rho_liq))
    dh_hem = st.h_mix - h_out_P2
    v_hem = float(np.sqrt(2.0 * dh_hem)) if dh_hem > 0 else v_sivi

    # Wallis ses hızı (Eq. 17) ve homojen Mach: homojen hız ile homojen void tutarlı kullanılır
    t1 = alpha2_S1 * rho2v + (1.0 - alpha2_S1) * rho2l
    t2 = alpha2_S1 / (rho2v * c2v ** 2) + (1.0 - alpha2_S1) / (rho2l * c2l ** 2)
    c_w = float(np.sqrt(1.0 / (t1 * t2)))
    M_dual = (G_dual / rho2_hom) / c_w
    if M_dual > 1.05:
        uyarilar.append(f"TUTARLILIK: Dual-SPC çıkışında homojen Mach = {M_dual:.2f} > 1. Akış gerçekte kritik; "
                        f"debinin P2'ye duyarlılığı modelin gösterdiğinden düşük olabilir.")

    return AkisSonucu(
        Cd_tekfaz=Cd, Cc=Cc, G_ideal=G_ideal, G_eff=G_eff, G_kal=G_kal,
        K_nurick=float(K), K_kritik=float(K_kr), kavitasyonlu=bool(kav), Cd_nurick=float(Cd_nur),
        P_hem_bogulma_pa=float(P_ch), hem_bogulmus=hem_chk,
        T2_sat_K=float(T2_sat), rho_2_liq=float(rho2l), rho_2_vap=float(rho2v),
        x_2=float(x2), alpha_2=float(alpha2), alpha_2_S1=float(alpha2_S1),
        S_2_zivi=S_zivi, S_2_aktif=S_act, rho_2_homojen=float(rho2_hom),
        r_kr_liq=r_kr_l, r_kr_vap=r_kr_v, sivi_bogulmus=liq_chk, buhar_bogulmus=vap_chk,
        v_jet_sivi=v_sivi, v_jet_hem=v_hem, c_2_wallis=c_w, Mach_dual_homojen=float(M_dual),
        kappa_dyer=kappa, tau_r_s=float(L_orifis_m / v_sivi), uyarilar=uyarilar
    )


def akis_op(st: FluidState, op: OperatingConfig, orif: OrificeConfig,
            kal: Optional[Dict[str, float]] = None, P2_pa: Optional[float] = None,
            Cd: Optional[float] = None, Cc: Optional[float] = None) -> AkisSonucu:
    return iki_fazli_akisi_coz(st, op.P2_pa if P2_pa is None else P2_pa,
                               orif.Cd_tekfaz if Cd is None else Cd,
                               orif.Cc if Cc is None else Cc,
                               orif.L_orifis_mm * 1e-3, op.slip_mode, kal)


# %% [5] ORİFİS BOYUTLANDIRMA
def boyut_hesapla_mm(m_dot: float, G: float, orif: OrificeConfig) -> float:
    """Hedef debi ve akı -> boyut [mm] (orifis çapı veya annulus aralığı)."""
    return float(orif.boyut_mm(m_dot / max(G, 1.0)))


@dataclass
class BoyutSonucu:
    d_mm: Dict[str, float]              # model -> boyut [mm] (orifis çapı veya annulus aralığı)
    A_mm2: Dict[str, float]             # model -> toplam akış alanı [mm2]
    d_tasarim_mm: float
    A_tasarim_mm2: float
    bant_min_mm: float
    bant_max_mm: float
    bant_min_model: str
    bant_max_model: str
    ilk_delme_mm: float                 # bandın küçük ucu, aşağı yuvarlanmış (orifis 0.05, annulus 0.01 mm)
    L_d: float                          # L / hidrolik çap
    Re_delik: float                     # hidrolik çapla
    D_dis_tasarim_mm: float             # annulus dış çapı (orifiste nan)
    duzgunsuzluk_pct: float             # annulus: eksantriklikten çevresel debi farkı ±% (orifiste 0)
    uyarilar: List[str]


def orifis_boyutlandir(m_dot: float, fl: AkisSonucu, st: FluidState, op: OperatingConfig,
                       orif: OrificeConfig) -> BoyutSonucu:
    d = {k: boyut_hesapla_mm(m_dot, G, orif) for k, G in fl.G_kal.items()}
    A = {k: m_dot / max(G, 1.0) * 1e6 for k, G in fl.G_kal.items()}
    bant = {k: d[k] for k in BANT_MODELLERI if not (k == 'nurick' and fl.K_nurick < 0.05)}
    k_min = min(bant, key=bant.get)
    k_max = max(bant, key=bant.get)
    d_t = d[op.tasarim_modeli]
    A_t = float(orif.alan_m2(d_t) * 1e6)
    adim = 0.01 if orif.annulus else 0.05
    ilk = float(np.floor(bant[k_min] / adim + 1e-9) * adim)
    Dh = orif.hidrolik_cap_m(d_t)
    L_d = orif.L_orifis_mm * 1e-3 / Dh
    Re = fl.G_kal[op.tasarim_modeli] * Dh / st.mu_liq
    uy: List[str] = []
    duz = 0.0
    if orif.annulus:
        duz = 100.0 * orif.eksantriklik_mm / d_t
        if L_d > 5.0:
            uy.append(f"Annulus L/Dh = {L_d:.1f} > 5: kanal sürtünmesi önemli, modellerde sürtünme yok. "
                      f"Cd_tekfaz düşük alınmalı ve testle ölçülmeli.")
        if bant[k_min] < 0.3:
            uy.append(f"Aralık {bant[k_min]:.2f} mm < 0.3 mm: işleme ve merkezleme hassasiyeti kritik.")
        uy.append(f"Eksantriklik ±{orif.eksantriklik_mm:.3f} mm -> çevresel debi farkı ~±%{duz:.1f} "
                  f"(tabaka bir tarafta kalın, diğerinde ince).")
        uy.append(f"Tolerans ±{orif.drill_tolerance_mm:.3f} mm aralığa uygulanır -> debide ~±%"
                  f"{100*orif.drill_tolerance_mm/d_t:.1f}.")
    elif L_d < 5.0 and orif.inlet_edge_type == 'sharp' and fl.kavitasyonlu:
        uy.append(f"L/d = {L_d:.1f} < 5, keskin giriş ve kavitasyon: hidrolik flip/yeniden tutunma bölgesi. "
                  f"Cd testle doğrulanmalı.")
    return BoyutSonucu(d, A, d_t, A_t, bant[k_min], bant[k_max], k_min, k_max, ilk, L_d, Re,
                       orif.D_dis_mm(d_t), duz, uy)


# %% [6] BESLEME HATTI
@dataclass
class BeslemeSonucu:
    v_hat_m_s: float
    v_manifold_m_s: float
    Re_hat: float
    f_darcy: float
    alan_orani_hat_orifis: float
    dP_tank_cikis_pa: float
    dP_vana_pa: float
    dP_boru_pa: float
    dP_fitting_pa: float
    dP_manifold_pa: float
    dP_toplam_pa: float
    P_vana_oncesi_pa: float
    P_vana_sonrasi_pa: float
    P1_efektif_pa: float
    alt_sogutma_manifold_bar: float
    dP_vana_bogulma_pa: float
    vana_kavitasyon: bool
    T_tank_max_K: float                 # manifoldda kaynama başlamadan izin verilen max tank sıcaklığı
    uyarilar: List[str]


def besleme_sistemi_coz(m_dot: float, st_tank: FluidState, feed: FeedSystemConfig,
                        A_orifis_tot_mm2: float) -> BeslemeSonucu:
    uy: List[str] = []
    rho, mu = st_tank.rho_liq, st_tank.mu_liq
    D = feed.D_hat_mm * 1e-3
    A_hat = np.pi * D ** 2 / 4.0
    v = m_dot / (rho * A_hat)
    Re = rho * v * D / mu
    eD = feed.eps_boru_mm * 1e-3 / D
    if Re < 2300.0:
        f = 64.0 / Re
    else:
        f = root_scalar(lambda ff: 1.0 / np.sqrt(ff) + 2.0 * np.log10(eD / 3.7 + 2.51 / (Re * np.sqrt(ff))),
                        bracket=[0.003, 0.2], method='brentq').root
    q = 0.5 * rho * v ** 2

    A_man = np.pi * feed.D_manifold_ic_mm * 1e-3 * feed.H_manifold_ic_mm * 1e-3   # radyal akış alanı
    v_man = m_dot / (rho * A_man)

    SG = rho / 999.0
    Q_m3h = m_dot / rho * 3600.0
    Kv = 0.865 * feed.Cv_ana_vana
    dP_tank = feed.K_tank_cikis * q
    dP_vana = SG * (Q_m3h / Kv) ** 2 * 1e5
    dP_boru = f * (feed.L_hat_m / D) * q
    dP_fit = feed.K_fitting_toplam * q
    dP_man = feed.K_manifold_giris * q
    dP_top = dP_tank + dP_vana + dP_boru + dP_fit + dP_man

    P_vo = st_tank.P_pa - dP_tank
    P_vs = P_vo - dP_vana
    P1 = P_vs - dP_boru - dP_fit - dP_man

    # Vana boğulmalı kavitasyon (ISA/IEC 60534): dP_kr = FL^2 (P1 - FF*Pv)
    FF = 0.96 - 0.28 * np.sqrt(st_tank.Pv_pa / P_CRIT)
    dP_kr = feed.FL_vana ** 2 * (P_vo - FF * st_tank.Pv_pa)
    vana_kav = dP_vana >= dP_kr

    alt_man = (P1 - st_tank.Pv_pa) / 1e5
    T_max = CP.PropsSI('T', 'P', P1, 'Q', 0, FLUID) if P1 < P_CRIT else T_CRIT

    if alt_man <= 0.0:
        uy.append(f"KRİTİK: Manifold basıncı ({P1/1e5:.2f} bar) Pv'nin ({st_tank.Pv_pa/1e5:.2f} bar) altında, hatta kaynama!")
    elif alt_man < 3.0:
        uy.append(f"Manifold alt-soğutma marjı düşük: {alt_man:.2f} bar (< 3 bar).")
    if vana_kav:
        uy.append(f"VANA KAVİTASYONU: dP_vana = {dP_vana/1e5:.2f} bar >= boğulma sınırı {dP_kr/1e5:.2f} bar.")
    if A_hat * 1e6 / max(A_orifis_tot_mm2, 1e-6) < 4.0:
        uy.append("Hat/orifis alan oranı < 4: hat hızı orifis akışını etkileyebilir.")
    if v > 10.0:
        uy.append(f"Hat hızı yüksek: {v:.1f} m/s.")

    return BeslemeSonucu(
        v_hat_m_s=v, v_manifold_m_s=v_man, Re_hat=Re, f_darcy=float(f),
        alan_orani_hat_orifis=A_hat * 1e6 / max(A_orifis_tot_mm2, 1e-6),
        dP_tank_cikis_pa=dP_tank, dP_vana_pa=dP_vana, dP_boru_pa=dP_boru, dP_fitting_pa=dP_fit,
        dP_manifold_pa=dP_man, dP_toplam_pa=dP_top, P_vana_oncesi_pa=P_vo, P_vana_sonrasi_pa=P_vs,
        P1_efektif_pa=P1, alt_sogutma_manifold_bar=alt_man, dP_vana_bogulma_pa=dP_kr,
        vana_kavitasyon=bool(vana_kav), T_tank_max_K=float(T_max), uyarilar=uy
    )


# %% [7] AYNA TERMAL MODELİ (YANMA + KAPANMA SONRASI SOAKBACK)
def _fdm_1d(L_m: float, t_s: float, T_init, T_gaz: float, h_gaz: float, T_n2o: float,
            h_n2o: float, malzeme: str, N: int = 41, dt: float = 0.05) -> Tuple[np.ndarray, float]:
    """1D örtük sonlu farklar. Düğüm 0 = gaz yüzü, düğüm N-1 = N2O yüzü.
    Döndürür: (son profil, süre boyunca N2O yüzünün maksimum sıcaklığı)."""
    dx = L_m / (N - 1)
    T = np.full(N, float(T_init)) if np.isscalar(T_init) else np.array(T_init, dtype=float)
    T_c_max = float(T[-1])
    n_steps = max(1, int(np.ceil(t_s / dt)))
    for _ in range(n_steps):
        k, rho, cp, _ = malzeme_ozellikleri_al(malzeme, float(np.mean(T)))
        Fo = (k / (rho * cp)) * dt / dx ** 2
        A = np.zeros((N, N))
        b = T.copy()
        Bi_g = h_gaz * dx / k
        A[0, 0] = 1.0 + 2.0 * Fo * (1.0 + Bi_g)
        A[0, 1] = -2.0 * Fo
        b[0] += 2.0 * Fo * Bi_g * T_gaz
        idx = np.arange(1, N - 1)
        A[idx, idx - 1] = -Fo
        A[idx, idx] = 1.0 + 2.0 * Fo
        A[idx, idx + 1] = -Fo
        Bi_c = h_n2o * dx / k
        A[-1, -2] = -2.0 * Fo
        A[-1, -1] = 1.0 + 2.0 * Fo * (1.0 + Bi_c)
        b[-1] += 2.0 * Fo * Bi_c * T_n2o
        T = np.linalg.solve(A, b)
        T_c_max = max(T_c_max, float(T[-1]))
    return T, T_c_max


def _depolanan_enerji_J(T_prof: np.ndarray, T_ref: float, L_m: float, A_m2: float, malzeme: str) -> float:
    x = np.linspace(0.0, L_m, len(T_prof))
    rc = np.array([malzeme_ozellikleri_al(malzeme, t)[1] * malzeme_ozellikleri_al(malzeme, t)[2] for t in T_prof])
    _trap = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')   # NumPy 1.x / 2.x uyumu
    return float(A_m2 * _trap(rc * (T_prof - T_ref), x))


@dataclass
class TermalSonuc:
    mat_etiket: str
    A_ayna_m2: float
    h_n2o_manifold: float
    h_n2o_giris: float
    h_n2o_delik: float
    h_n2o_efektif: float
    fsolve_ok: bool
    Tw_gaz_kararli_K: float
    Tw_n2o_kararli_K: float
    Tw_gaz_tb_K: float                  # yanma sonu (t_yanma), nominal sınır şartı
    Tw_n2o_tb_K: float
    Q_tb_W: float
    grid: List[Tuple[float, float, float, float, float, float]]   # (Tg, hg, Twg°C, Twn°C, Q kW, dT_bulk K)
    dT_bulk_nom_K: float
    dT_bulk_max_K: float
    T_sat_manifold_K: float
    T_onb_K: float
    onb_yanma: bool
    q_ortalama_W_m2: float
    # Kapanma sonrası
    E_depolanan_nom_J: float
    E_depolanan_max_J: float
    E_hapsolan_kaynama_J: float
    m_hapsolan_kg: float
    Tw_n2o_soak_max_K: float            # kapanma sonrası N2O yüzü maksimumu (h_durgun ile, en kötü zarf)
    T_denge_kuru_nom_K: float           # hapsolan N2O kaynadıktan sonra ayna denge sıcaklığı (nominal)
    T_denge_kuru_K: float               # aynı, en kötü zarf
    hapsolan_tamamen_kaynar: bool
    durum_ozeti: str


def termal_coz(m_dot: float, st_in: FluidState, feed: FeedSystemConfig, fd: BeslemeSonucu,
               d_mm: float, orif: OrificeConfig, th: ThermalConfig) -> TermalSonuc:
    """d_mm: orifis çapı veya annulus aralığı. Kanal duvarı = ıslak çevre x boy, h hidrolik çapla."""
    L = orif.L_orifis_mm * 1e-3
    Dh = orif.hidrolik_cap_m(d_mm)
    cevre = orif.islak_cevre_m(d_mm)
    A_akis = float(orif.alan_m2(d_mm))
    T_n2o = st_in.T_K
    P_man = min(fd.P1_efektif_pa, P_CRIT - 5e4)

    A_ayna = np.pi / 4.0 * ((th.D_ayna_mm * 1e-3) ** 2 - (th.D_merkez_mm * 1e-3) ** 2)
    A1 = max(A_ayna - A_akis, 1e-6)
    L_giris = min(Dh, 0.3 * L)
    A2 = cevre * L_giris
    A3 = cevre * max(L - L_giris, 1e-4)

    # Galeri: paralel levha aralığı, hidrolik çap = 2H
    Dh_man = 2.0 * feed.H_manifold_ic_mm * 1e-3
    Re_man = st_in.rho_liq * max(fd.v_manifold_m_s, 0.1) * Dh_man / st_in.mu_liq
    if Re_man > 2300.0:
        Nu_man = 0.023 * Re_man ** 0.8 * st_in.Pr_liq ** 0.4
    else:
        Nu_man = 7.54                                   # paralel levha, laminer, sabit T
    h1 = Nu_man * st_in.k_liq / Dh_man

    Re_delik = (m_dot / A_akis) * Dh / st_in.mu_liq
    Nu_d = float(ht.Nu_conv_internal(Re=max(Re_delik, 2500.0), Pr=st_in.Pr_liq, eD=1e-4, Di=Dh, x=L))
    h3 = Nu_d * st_in.k_liq / Dh
    h2 = 1.35 * h3                                      # giriş bölgesi artışı (yaklaşık)
    UA = h1 * A1 + h2 * A2 + h3 * A3
    h_eff = UA / A_ayna

    # Kararlı hal (yalnızca referans)
    def denge(v):
        Tg, Tc = v
        k_m = malzeme_ozellikleri_al(th.malzeme, 0.5 * (Tg + Tc))[0]
        q_g = th.h_gaz_nominal * A_ayna * (th.T_gaz_eff_nom_K - Tg)
        q_k = k_m * A_ayna / L * (Tg - Tc)
        q_c = UA * (Tc - T_n2o)
        return [q_g - q_k, q_k - q_c]
    sol, _, ier, _ = fsolve(denge, [800.0, T_n2o + 30.0], full_output=True)

    # Yanma sırasında 3x3 sınır şartı zarfı
    h_ref = CP.PropsSI('H', 'P', P_man, 'T', T_n2o, FLUID)

    def _dT_bulk(Q_W: float, f_mix: float) -> float:
        return CP.PropsSI('T', 'H', h_ref + f_mix * Q_W / m_dot, 'P', P_man, FLUID) - T_n2o

    grid = []
    prof_nom = prof_max = None
    Twg_tb = Twn_tb = Q_tb = 0.0
    for Tg in (th.T_gaz_eff_low_K, th.T_gaz_eff_nom_K, th.T_gaz_eff_high_K):
        for hg in (th.h_gaz_low, th.h_gaz_nominal, th.h_gaz_high):
            prof, _ = _fdm_1d(L, th.t_yanma_s, T_n2o, Tg, hg, T_n2o, h_eff, th.malzeme)
            Q = UA * (prof[-1] - T_n2o)
            grid.append((Tg, hg, prof[0] - 273.15, prof[-1] - 273.15, Q / 1e3, _dT_bulk(Q, th.f_mix_nominal)))
            if Tg == th.T_gaz_eff_nom_K and hg == th.h_gaz_nominal:
                prof_nom, Twg_tb, Twn_tb, Q_tb = prof, prof[0], prof[-1], Q
            if Tg == th.T_gaz_eff_high_K and hg == th.h_gaz_high:
                prof_max = prof
    Q_max = max(r[4] for r in grid) * 1e3
    dT_nom = _dT_bulk(Q_tb, th.f_mix_nominal)
    dT_max = _dT_bulk(Q_max, 1.0)                       # üst sınır: tam karışım + en yüksek ısı

    # Kaynama başlangıcı (Davis-Anderson), galeri duvarında, alt-soğutulmuş zorlanmış taşınım
    T_sat = CP.PropsSI('T', 'P', P_man, 'Q', 0, FLUID)
    rho_v = CP.PropsSI('D', 'P', P_man, 'Q', 1, FLUID)
    rho_ls = CP.PropsSI('D', 'P', P_man, 'Q', 0, FLUID)
    h_fg = CP.PropsSI('H', 'P', P_man, 'Q', 1, FLUID) - CP.PropsSI('H', 'P', P_man, 'Q', 0, FLUID)
    _, k_ls, _, sig = _n2o_transport(round(T_sat - 1e-3, 2), round(P_man, -3))
    q_ort = max(Q_tb / A_ayna, 1.0)
    T_onb = T_sat + float(np.sqrt(8.0 * sig * T_sat * q_ort / (k_ls * h_fg * rho_v)))
    onb = Twn_tb > T_onb

    # Kapanma sonrası soakback: gaz yüzü adyabatik (oda duvarlarından ışınım ihmal), N2O durgun
    _, Twn_soak_nom = _fdm_1d(L, th.t_kapanma_sonrasi_s, prof_nom, 0.0, 0.0, T_n2o, th.h_n2o_durgun, th.malzeme)
    _, Twn_soak_max = _fdm_1d(L, th.t_kapanma_sonrasi_s, prof_max, 0.0, 0.0, T_n2o, th.h_n2o_durgun, th.malzeme)
    E_nom = _depolanan_enerji_J(prof_nom, T_n2o, L, A_ayna, th.malzeme)
    E_max = _depolanan_enerji_J(prof_max, T_n2o, L, A_ayna, th.malzeme)
    m_hap = rho_ls * th.V_hapsolan_cm3 * 1e-6
    cp_l = CP.PropsSI('C', 'P', P_man, 'T', min(T_n2o, T_sat - 0.01), FLUID)
    E_kay = m_hap * (cp_l * max(T_sat - T_n2o, 0.0) + h_fg)
    tam_kaynar = E_max > E_kay
    k_m, rho_m, cp_m, etiket = malzeme_ozellikleri_al(th.malzeme, float(np.mean(prof_max)))
    m_metal = rho_m * A_ayna * L
    T_denge = T_n2o + max(E_max - E_kay, 0.0) / (m_metal * cp_m)
    cp_nom = malzeme_ozellikleri_al(th.malzeme, float(np.mean(prof_nom)))[2]
    T_denge_nom = T_n2o + max(E_nom - E_kay, 0.0) / (m_metal * cp_nom)

    T_kontrol = max(Twn_tb, Twn_soak_max, T_denge, max(r[3] for r in grid) + 273.15)
    if T_kontrol > th.T_wall_hazard_K:
        durum = (f"KRİTİK: N2O ıslak yüzey {T_kontrol-273.15:.0f} °C > {th.T_wall_hazard_K-273.15:.0f} °C eşiği "
                 f"(en kötü zarf; nominal kapanma sonrası denge {T_denge_nom-273.15:.0f} °C). Kapanmada N2 purge şart.")
    elif T_kontrol > th.T_wall_caution_K:
        durum = (f"UYARI: N2O ıslak yüzey {T_kontrol-273.15:.0f} °C > {th.T_wall_caution_K-273.15:.0f} °C eşiği "
                 f"(en kötü zarf). Kapanmada N2 purge önerilir.")
    elif tam_kaynar:
        durum = ("SOAKBACK: Hapsolan N2O kapanmadan sonra tamamen buharlaşıyor (ayna kuru kalıyor). "
                 "Kapanmada N2 purge önerilir.")
    elif onb:
        durum = f"BİLGİ: Yanma sonunda galeri duvarında kaynama başlangıcı (Tw = {Twn_tb-273.15:.1f} °C > T_ONB = {T_onb-273.15:.1f} °C)."
    else:
        durum = "GÜVENLİ: Yanma ve kapanma sonrası N2O yüzeyi eşiklerin altında."

    return TermalSonuc(
        mat_etiket=etiket, A_ayna_m2=A_ayna, h_n2o_manifold=h1, h_n2o_giris=h2, h_n2o_delik=h3,
        h_n2o_efektif=h_eff, fsolve_ok=(ier == 1), Tw_gaz_kararli_K=float(sol[0]), Tw_n2o_kararli_K=float(sol[1]),
        Tw_gaz_tb_K=float(Twg_tb), Tw_n2o_tb_K=float(Twn_tb), Q_tb_W=float(Q_tb), grid=grid,
        dT_bulk_nom_K=float(dT_nom), dT_bulk_max_K=float(dT_max),
        T_sat_manifold_K=float(T_sat), T_onb_K=float(T_onb), onb_yanma=bool(onb), q_ortalama_W_m2=float(q_ort),
        E_depolanan_nom_J=E_nom, E_depolanan_max_J=E_max, E_hapsolan_kaynama_J=float(E_kay),
        m_hapsolan_kg=float(m_hap), Tw_n2o_soak_max_K=float(Twn_soak_max), T_denge_kuru_nom_K=float(T_denge_nom),
        T_denge_kuru_K=float(T_denge),
        hapsolan_tamamen_kaynar=bool(tam_kaynar), durum_ozeti=durum
    )


# %% [8] BAĞLAŞIK ÇÖZÜM (BESLEME <-> ORİFİS <-> TERMAL)
@dataclass
class Cozum:
    yakinsadi: bool
    iterasyon: int
    st_tank: FluidState
    fl_adyabatik: AkisSonucu            # kayıpsız/ısısız referans (tank durumu doğrudan orifiste)
    boy_adyabatik: BoyutSonucu
    st_in: FluidState                   # orifis girişi (hat kaybı + ısınma sonrası)
    fl: AkisSonucu
    boy: BoyutSonucu
    fd: BeslemeSonucu
    th: TermalSonuc


def baglasik_coz(op: OperatingConfig, orif: OrificeConfig, feed: FeedSystemConfig, th: ThermalConfig,
                 kal: Optional[Dict[str, float]] = None, max_iter: int = 15, tol_mm: float = 1e-4) -> Cozum:
    m_dot = op.m_dot_hedef
    st_tank = durum_op(op.P_tank_pa, op.T_tank_K, op)
    fl_ad = akis_op(st_tank, op, orif, kal)
    boy_ad = orifis_boyutlandir(m_dot, fl_ad, st_tank, op, orif)

    boy, T_in = boy_ad, op.T_tank_K
    yak, it = False, 0
    for it in range(1, max_iter + 1):
        fd = besleme_sistemi_coz(m_dot, st_tank, feed, boy.A_tasarim_mm2)
        if fd.P1_efektif_pa <= op.P2_pa + 1e5:
            raise ValueError(f"Hat kayıpları sonrası P1 ({fd.P1_efektif_pa/1e5:.2f} bar) P2'ye çok yakın.")
        st_in = durum_op(fd.P1_efektif_pa, T_in, op)
        fl = akis_op(st_in, op, orif, kal)
        boy_yeni = orifis_boyutlandir(m_dot, fl, st_in, op, orif)
        thr = termal_coz(m_dot, st_in, feed, fd, boy_yeni.d_tasarim_mm, orif, th)
        T_in_yeni = op.T_tank_K + thr.dT_bulk_nom_K
        hata = abs(boy_yeni.d_tasarim_mm - boy.d_tasarim_mm)
        boy = boy_yeni
        if hata < tol_mm and abs(T_in_yeni - T_in) < 1e-3:
            yak = True
            break
        T_in = T_in_yeni
    return Cozum(yak, it, st_tank, fl_ad, boy_ad, st_in, fl, boy, fd, thr)


# %% [9] KALİBRASYON (KENDİ SOĞUK AKIŞ TESTLERİNİZ)
def kalibrasyon_hesapla(testler: List[KalibrasyonTesti], op: OperatingConfig,
                        orif: OrificeConfig) -> Dict[str, Tuple[float, float, int]]:
    """Her model için k = ölçülen / tahmin. Döndürür: {model: (ortalama k, std k, n)}.
    Testler, tasarım koşuluna (alt-soğutma, P2) olabildiğince yakın yapılmalıdır."""
    oran: Dict[str, List[float]] = {k: [] for k in MODEL_ADLARI}
    for t in testler:
        st = durum_op(t.P1_bar * 1e5, t.T1_K, op)
        fl = akis_op(st, op, orif, None, P2_pa=t.P2_bar * 1e5)
        if t.D_ic_mm:
            geo = replace(orif, geometri='annulus', D_ic_mm=t.D_ic_mm)
        else:
            geo = replace(orif, geometri='orifis', N_orifis=t.N_orifis)
        A = float(geo.alan_m2(t.d_mm))
        for k, G in fl.G_eff.items():
            if G > 0:
                oran[k].append(t.mdot_olculen / (G * A))
    sonuc = {}
    for k, v in oran.items():
        if v:
            arr = np.array(v)
            sonuc[k] = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else float('nan'), len(arr))
    return sonuc


# %% [10] İLERİ YÖNLÜ MATKAP TABLOSU (P2 SABİT)
@dataclass
class MatkapSatiri:
    d_mm: float
    mdot: Dict[str, float]              # model -> debi [kg/s] (hat kaybı bağlaşık)
    mdot_tol_eksi: float                # tasarım modeli, d - tol
    mdot_tol_arti: float
    dlnm_dlnP2: float                   # tasarım modeli, P2 duyarlılığı (0 = tam boğulmuş)
    not_str: str


def matkap_tablosu_coz(sol: Cozum, op: OperatingConfig, orif: OrificeConfig, feed: FeedSystemConfig,
                       kal: Optional[Dict[str, float]], matkaplar: List[float]) -> List[MatkapSatiri]:
    m_hedef = op.m_dot_hedef
    T_in = sol.st_in.T_K
    modeller = ('spi',) + BANT_MODELLERI
    satirlar = []
    for d in matkaplar:
        A = float(orif.alan_m2(d))
        mdot = {}
        st_son = sol.st_in
        kaynayan = []
        for m in modeller:
            md = m_hedef
            try:
                for j in range(30):
                    fd = besleme_sistemi_coz(md, sol.st_tank, feed, A * 1e6)
                    st = durum_op(fd.P1_efektif_pa, T_in, op)
                    md_yeni = akis_op(st, op, orif, kal).G_kal[m] * A
                    if abs(md_yeni - md) < 1e-5:
                        md = md_yeni
                        break
                    md = md_yeni if j < 3 else 0.5 * (md + md_yeni)   # yakınsama için sönümleme
            except ValueError:
                md = float('nan')                  # bu debide hat kayıpları manifoldu Pv'nin altına indiriyor
                kaynayan.append(m)
            mdot[m] = md
            if m == op.tasarim_modeli and not np.isnan(md):
                st_son = st
        mt = mdot[op.tasarim_modeli]
        tol = orif.drill_tolerance_mm
        m_eksi = mt * float(orif.alan_m2(d - tol)) / A
        m_arti = mt * float(orif.alan_m2(d + tol)) / A
        dln = float('nan')
        if not np.isnan(mt):
            dP2 = 2.0e5
            Gp = akis_op(st_son, op, orif, kal, P2_pa=op.P2_pa + dP2).G_kal[op.tasarim_modeli]
            Gm = akis_op(st_son, op, orif, kal, P2_pa=op.P2_pa - dP2).G_kal[op.tasarim_modeli]
            G0 = akis_op(st_son, op, orif, kal).G_kal[op.tasarim_modeli]
            dln = (Gp - Gm) / G0 * op.P2_pa / (2.0 * dP2)
        bant_vals = [mdot[m] for m in BANT_MODELLERI if not np.isnan(mdot[m])]
        icinde = bool(bant_vals) and min(bant_vals) <= m_hedef <= max(bant_vals)
        not_str = (f"Tasarım %{(mt/m_hedef-1)*100:+.1f}" if not np.isnan(mt) else "Tasarım: HAT KAYNAR")
        not_str += " | hedef bant içinde" if icinde else ""
        if kaynayan:
            not_str += f" | hat kaynar: {','.join(kaynayan)}"
        satirlar.append(MatkapSatiri(d, mdot, m_eksi, m_arti, dln, not_str))
    return satirlar


# %% [11] DUYARLILIK (TORNADO) VE MONTE CARLO
@dataclass
class UQSonuc:
    tornado: Dict[str, Tuple[float, float, float]]   # ad -> (d_min, d_max, fark) [mm]
    dlnG_dlnP1: float
    dG_dT1_pct_K: float
    d_hw_P10: float
    d_hw_P50: float
    d_hw_P90: float
    d_karma_P10: float
    d_karma_P50: float
    d_karma_P90: float
    ilk_delme_mm: float
    mdot_ilk_hw_P10: float
    mdot_ilk_hw_P50: float
    mdot_ilk_hw_P90: float
    mdot_ilk_karma_P10: float
    mdot_ilk_karma_P50: float
    mdot_ilk_karma_P90: float
    notlar: List[str]


def belirsizlik_coz(sol: Cozum, op: OperatingConfig, orif: OrificeConfig,
                    kal: Optional[Dict[str, float]], n: int = 4000, seed: int = 42) -> UQSonuc:
    rng = np.random.default_rng(seed)
    m_dot = op.m_dot_hedef
    tm = op.tasarim_modeli
    st0 = sol.st_in
    P0, T0 = st0.P_pa, st0.T_K
    fl0 = sol.fl
    d0 = sol.boy.d_tasarim_mm
    notlar: List[str] = []

    def _G(P, T, op_=op, Cd=None, Cc=None) -> Dict[str, float]:
        st = durum_op(P, T, op_)
        return akis_op(st, op_, orif, kal, Cd=Cd, Cc=Cc).G_kal

    def _d(G: float) -> float:
        return boyut_hesapla_mm(m_dot, G, orif)

    # Türevler (tüm modeller tek seferde)
    dPs, dTs = 1.0e5, 0.5
    Gp, Gm = _G(P0 + dPs, T0), _G(P0 - dPs, T0)
    try:
        GTp = _G(P0, T0 + dTs)
        GTm = _G(P0, T0 - dTs)
        dT_den = 2.0 * dTs
    except ValueError:
        GTp, GTm, dT_den = fl0.G_kal, _G(P0, T0 - dTs), dTs
        notlar.append("T1 + 0.5 K faz sınırını aşıyor: sıcaklık türevi tek taraflı.")
    aP = {k: (Gp[k] - Gm[k]) / (2.0 * dPs) / fl0.G_kal[k] for k in fl0.G_kal}   # 1/Pa
    aT = {k: (GTp[k] - GTm[k]) / dT_den / fl0.G_kal[k] for k in fl0.G_kal}      # 1/K

    # Tornado (tasarım modeli için tam yeniden hesap)
    torn: Dict[str, Tuple[float, float, float]] = {}

    def _ekle(ad: str, d1: float, d2: float):
        torn[ad] = (min(d1, d2), max(d1, d2), abs(d2 - d1))

    Cd0 = orif.Cd_tekfaz
    Cd_lo, Cd_hi = max(Cd0 - CD_BELIRSIZLIK, 0.05), min(Cd0 + CD_BELIRSIZLIK, 1.0)
    _ekle(f"Cd tek faz ({Cd_lo:.2f} .. {Cd_hi:.2f})", _d(_G(P0, T0, Cd=Cd_hi)[tm]), _d(_G(P0, T0, Cd=Cd_lo)[tm]))
    Cc0 = orif.Cc
    _ekle(f"Nurick Cc ({Cc0-0.04:.2f} .. {Cc0+0.04:.2f})",
          _d(_G(P0, T0, Cc=Cc0 + 0.04)[tm]), _d(_G(P0, T0, Cc=Cc0 - 0.04)[tm]))
    _ekle("Manifold basıncı P1 (±3 bar)", _d(_G(P0 + 3e5, T0)[tm]), _d(_G(P0 - 3e5, T0)[tm]))
    try:
        _ekle("N2O sıcaklığı T1 (±2.5 K)", _d(_G(P0, T0 - 2.5)[tm]), _d(_G(P0, T0 + 2.5)[tm]))
    except ValueError as e:
        notlar.append(f"T1 + 2.5 K: {e}")
    diger_vap = 'P1_sat' if op.vap_ref_mode == 'T1_sat' else 'T1_sat'
    _ekle(f"Buhar referansı ({op.vap_ref_mode} / {diger_vap})", d0, _d(_G(P0, T0, replace(op, vap_ref_mode=diger_vap))[tm]))
    diger_slip = 'homogeneous' if op.slip_mode == 'zivi' else 'zivi'
    _ekle(f"Kayma oranı ({op.slip_mode} / {diger_slip})", d0, _d(_G(P0, T0, replace(op, slip_mode=diger_slip))[tm]))
    diger_exp = 'cp_cv' if op.exponent_mode == 'real_ns' else 'real_ns'
    _ekle(f"İzentropik üs ({op.exponent_mode} / {diger_exp})", d0, _d(_G(P0, T0, replace(op, exponent_mode=diger_exp))[tm]))
    _ekle("Hat kaybı + termal (kayıpsız / bağlaşık)", sol.boy_adyabatik.d_tasarim_mm, d0)
    _ekle(f"Model seçimi ({sol.boy.bant_min_model} .. {sol.boy.bant_max_model})", sol.boy.bant_min_mm, sol.boy.bant_max_mm)
    torn = dict(sorted(torn.items(), key=lambda kv: kv[1][2], reverse=True))

    # Monte Carlo (doğrusallaştırılmış, model başına kendi türevleriyle)
    Cd_s = rng.uniform(Cd_lo, Cd_hi, n)
    Cc_s = rng.uniform(Cc0 - 0.04, Cc0 + 0.04, n)
    dP_s = rng.normal(0.0, 1.5e5, n)
    dT_s = rng.uniform(-2.5, 2.5, n)
    K = fl0.K_nurick

    def _G_ornek(model: str, idx) -> np.ndarray:
        g = fl0.G_kal[model] * (1.0 + aP[model] * dP_s[idx] + aT[model] * dT_s[idx])
        if model == 'nurick':
            f = np.minimum(Cd_s[idx], Cc_s[idx] * np.sqrt(max(K, 0))) / min(Cd0, Cc0 * np.sqrt(max(K, 0)))
        else:
            f = Cd_s[idx] / Cd0
        return np.maximum(g * f, 1000.0)

    tum = np.arange(n)
    G_hw = _G_ornek(tm, tum)
    bant = [m for m in BANT_MODELLERI if not (m == 'nurick' and K < 0.05)]
    secim = rng.integers(0, len(bant), n)
    G_karma = np.empty(n)
    for i, m in enumerate(bant):
        idx = np.where(secim == i)[0]
        G_karma[idx] = _G_ornek(m, idx)

    def _dd(G):
        return orif.boyut_mm(m_dot / G)

    d_hw, d_karma = _dd(G_hw), _dd(G_karma)
    ilk = sol.boy.ilk_delme_mm
    d_ilk = ilk + rng.normal(0.0, orif.drill_tolerance_mm / 2.0, n)
    A_ilk = orif.alan_m2(d_ilk)
    m_hw, m_karma = G_hw * A_ilk, G_karma * A_ilk
    p = lambda a, q: float(np.percentile(a, q))
    notlar.append("MC, belirsizlikleri doğrusal türevlerle yayar; büyük sapmalarda (özellikle T1) yaklaşıktır.")

    return UQSonuc(
        tornado=torn, dlnG_dlnP1=float(aP[tm] * P0), dG_dT1_pct_K=float(aT[tm] * 100.0),
        d_hw_P10=p(d_hw, 10), d_hw_P50=p(d_hw, 50), d_hw_P90=p(d_hw, 90),
        d_karma_P10=p(d_karma, 10), d_karma_P50=p(d_karma, 50), d_karma_P90=p(d_karma, 90),
        ilk_delme_mm=ilk,
        mdot_ilk_hw_P10=p(m_hw, 10), mdot_ilk_hw_P50=p(m_hw, 50), mdot_ilk_hw_P90=p(m_hw, 90),
        mdot_ilk_karma_P10=p(m_karma, 10), mdot_ilk_karma_P50=p(m_karma, 50), mdot_ilk_karma_P90=p(m_karma, 90),
        notlar=notlar
    )


# %% [12] MAKALE KARŞILAŞTIRMASI (EUCASS 2023-728, SIVI TEST 1)
@dataclass
class VVSatiri:
    model: str
    kod_real_ns: float
    kod_cp_cv: float
    makale_modeli: Optional[float]
    deney: float


@dataclass
class VVSonuc:
    P1_bar: float
    T1_K: float
    P2_bar: float
    dP0_hata: float
    satirlar: List[VVSatiri]
    S2: float


def vv_makale_karsilastirmasi() -> VVSonuc:
    """
    Makale koşulu: doymuş sıvı N2O, d = 1 mm, Cd = 0.7, atmosfere boşaltma.
    Test başına sıcaklık makalede verilmediği için T1 = 293 K varsayılır.
    'Makale modeli' değerleri Tablo 2 (Test 1) ve Şekil 5'ten okunmuştur; tablodaki 'HEM'
    sütunu Şekil 5'teki HEMc eğrisine karşılık gelir. Deney (Tablo 2, Test 1) = 0.018 kg/s.
    Bu bir doğrulama DEĞİL, uygulamanın makaleyle tutarlılık kontrolüdür; asıl ölçüt
    deneye göre sapmadır ve yalnızca makalenin (doymuş, P2 = 1 atm) rejimini temsil eder.
    """
    T1 = 293.0
    P1 = CP.PropsSI('P', 'T', T1, 'Q', 0, FLUID)
    P2 = 101325.0
    A = np.pi * (1.0e-3) ** 2 / 4.0
    sonuc = {}
    S2 = 0.0
    for mode in ('real_ns', 'cp_cv'):
        st = termodinamik_durum_coz(P1, T1 - 1e-4, exponent_mode=mode, vap_ref_mode='T1_sat')
        fl = iki_fazli_akisi_coz(st, P2, 0.70, CC_KENAR['sharp'], 0.005, 'zivi')
        sonuc[mode] = {k: v * A for k, v in fl.G_eff.items()}
        S2 = fl.S_2_zivi
    makale = {'spi': 0.0495, 'spc_liq': 0.042, 'hem_lokal': 0.003, 'hem_c': 0.017,
              'dyer': 0.026, 'fml': 0.022, 'dual_spc': 0.013, 'nurick': None, 'dyer_comp': None}
    satirlar = [VVSatiri(k, sonuc['real_ns'][k], sonuc['cp_cv'][k], makale.get(k), 0.018) for k in MODEL_ADLARI]

    st_t = termodinamik_durum_coz(62e5, 293.1)
    G_spc, _, _ = _spc_ideal_flux(st_t.rho_liq, 62e5, 62e5 - 2000.0, st_t.n_active_liq)
    G_spi = np.sqrt(2.0 * st_t.rho_liq * 2000.0)
    return VVSonuc(P1 / 1e5, T1, P2 / 1e5, float(abs(G_spc - G_spi) / G_spi), satirlar, S2)


# %% [13] RAPOR
def _cizgi(c: str = '-', n: int = 104) -> str:
    return c * n


def rapor_yazdir(op: OperatingConfig, orif: OrificeConfig, feed: FeedSystemConfig, th: ThermalConfig,
                 vv: VVSonuc, sol: Cozum, matkap: List[MatkapSatiri], uq: UQSonuc,
                 kal_ozet: Dict[str, Tuple[float, float, int]], vv_yazdir: bool = True) -> None:
    st, fl, b, fd, t = sol.st_tank, sol.fl, sol.boy, sol.fd, sol.th
    bad = orif.boyut_adi
    BAD = 'ARALIK' if orif.annulus else 'ÇAP'
    geo_str = (f"ANNULUS (iç çap {orif.D_ic_mm:.2f} mm)" if orif.annulus else f"ORİFİS (N = {orif.N_orifis})")
    si = sol.st_in
    tm = op.tasarim_modeli
    P2b = op.P2_pa / 1e5
    print("\n" + _cizgi('='))
    print(f"   N2O ENJEKTÖR AKIŞ / BESLEME / AYNA TERMAL RAPORU  (V7.2)  -  {geo_str}")
    print(_cizgi('='))
    print(f"  Hedef debi: RPA {op.m_dot_rpa:.5f} kg/s / eta_atom {op.eta_atomizasyon:.2f} = {op.m_dot_hedef:.5f} kg/s"
          f" | {geo_str} | L = {orif.L_orifis_mm:.1f} mm | P2 = {P2b:.1f} bar | yanma {th.t_yanma_s:.1f} s")
    print(f"  Tasarım (referans) modeli: {MODEL_ADLARI[tm]} | kalibrasyon: "
          f"{'VAR (' + str(len(kal_ozet)) + ' model)' if kal_ozet else 'YOK (tüm çarpanlar 1.0)'}")

    if vv_yazdir:
        print(f"\n[0] MAKALE TUTARLILIK KONTROLÜ (EUCASS 2023-728, sıvı Test 1; T1 = {vv.T1_K:.0f} K varsayımı, "
              f"P1 = {vv.P1_bar:.2f} bar -> {vv.P2_bar:.3f} bar, d = 1 mm, Cd = 0.70)")
        print(_cizgi())
        print(f"  dP -> 0 limitinde SPC = SPI: bağıl fark {vv.dP0_hata:.1e} {'[OK]' if vv.dP0_hata < 5e-3 else '[HATA]'}"
              f" | Çıkış Zivi S2 = {vv.S2:.2f}")
        print(f"  {'Model':<44}| {'Kod real_ns':>11} | {'Kod cp_cv':>10} | {'Makale':>7} | {'Sapma(real)':>11} | {'Deneye göre':>11}")
        for r in vv.satirlar:
            mk = f"{r.makale_modeli:.4f}" if r.makale_modeli else "   -  "
            sap = f"%{(r.kod_real_ns/r.makale_modeli-1)*100:+.0f}" if r.makale_modeli else "-"
            dny = "geçersiz" if r.model == 'nurick' else f"%{(r.kod_real_ns/r.deney-1)*100:+.0f}"
            print(f"  {MODEL_ADLARI[r.model]:<44}| {r.kod_real_ns:>11.4f} | {r.kod_cp_cv:>10.4f} | {mk:>7} | {sap:>11} | {dny:>11}")
        print("  Not: Bu tablo yalnızca makale rejimini (doymuş, P2 = 1 atm) temsil eder; 27 bar karşı basınç için")
        print("       geçerlilik kanıtı DEĞİLDİR. Dual-SPC'de makale değerine cp_cv modu daha yakındır.")

    print(f"\n[1] GİRİŞ DURUMU")
    print(_cizgi())
    print(f"  Tank: {st.P_pa/1e5:.2f} bar, {st.T_K-273.15:.2f} °C -> {st.faz} | Pv(T1) = {st.Pv_pa/1e5:.2f} bar"
          f" | tank alt-soğutma = {st.alt_sogutma_pa/1e5:.2f} bar")
    print(f"  Orifis girişi: {si.P_pa/1e5:.2f} bar, {si.T_K-273.15:.2f} °C | alt-soğutma = {si.alt_sogutma_pa/1e5:.2f} bar"
          f" | rho_l = {si.rho_liq:.1f} kg/m3 | ns_l = {si.ns_liq:.2f} | Cp/Cv = {si.gamma_liq:.2f}")
    va = si.vap_active
    print(f"  Aktif buhar referansı: {va.label} | rho_v = {va.rho:.1f} kg/m3 | n = {va.n_active:.3f}"
          f" (ns = {va.ns_real:.3f}, Cp/Cv = {va.gamma_cp_cv:.2f})")
    for w in dict.fromkeys(si.uyarilar or st.uyarilar):
        print(f"    [!] {w}")

    print(f"\n[2] ÇIKIŞ KANALI AKIŞ REJİMİ (P2 = {P2b:.1f} bar)")
    print(_cizgi())
    print(f"  Nurick K = (P1-Pv)/(P1-P2) = {fl.K_nurick:.3f} | K_kr = (Cd/Cc)^2 = {fl.K_kritik:.2f} -> "
          f"{'KAVİTASYONLU' if fl.kavitasyonlu else 'kavitasyonsuz'} (Cc = {fl.Cc:.2f}, Cd_kav = {fl.Cd_nurick:.3f})")
    print(f"  HEM boğulma basıncı = {fl.P_hem_bogulma_pa/1e5:.2f} bar -> {'BOĞULMUŞ' if fl.hem_bogulmus else 'boğulmamış'}"
          f" | Dyer kappa = {fl.kappa_dyer:.3f} | kalma süresi = {fl.tau_r_s*1e6:.0f} us")
    print(f"  Sıvı SPC: r = {op.P2_pa/si.P_pa:.3f} vs r_kr = {fl.r_kr_liq:.3f} ({'boğulmuş' if fl.sivi_bogulmus else 'boğulmamış'})"
          f" | Buhar SPC: r_kr = {fl.r_kr_vap:.3f} ({'boğulmuş' if fl.buhar_bogulmus else 'boğulmamış'})")
    print(f"  Çıkış: T2_sat = {fl.T2_sat_K-273.15:.1f} °C | x2 = %{fl.x_2*100:.1f} | void (S={fl.S_2_aktif:.2f}) = "
          f"%{fl.alpha_2*100:.1f} | void (S=1) = %{fl.alpha_2_S1*100:.1f}")
    print(f"  Wallis c2 = {fl.c_2_wallis:.0f} m/s | Dual-SPC homojen Mach = {fl.Mach_dual_homojen:.2f}")
    for w in fl.uyarilar + b.uyarilar:
        print(f"    [!] {w}")

    print(f"\n[3] MODEL AKILARI VE GEREKEN {BAD}LAR (hedef {op.m_dot_hedef:.4f} kg/s, {geo_str})")
    print(_cizgi())
    print(f"  {'Model':<46}| {'G [kg/m2s]':>11} | {'A [mm2]':>8} | {bad + ' [mm]':>11} | Rol")
    for k in MODEL_ADLARI:
        rol = "BANT" if k in BANT_MODELLERI else "referans"
        if k == tm:
            rol += " + TASARIM"
        print(f"  {MODEL_ADLARI[k]:<46}| {fl.G_kal[k]:>11.0f} | {b.A_mm2[k]:>8.2f} | {b.d_mm[k]:>11.3f} | {rol}")
    print(f"  >> {BAD} BANDI: {b.bant_min_mm:.3f} mm ({b.bant_min_model}) .. {b.bant_max_mm:.3f} mm ({b.bant_max_model})"
          f" | alan {b.A_mm2[b.bant_min_model]:.1f} .. {b.A_mm2[b.bant_max_model]:.1f} mm2")
    if orif.annulus:
        print(f"  >> İLK İŞLEME ÖNERİSİ: aralık {b.ilk_delme_mm:.2f} mm -> dış çap {orif.D_dis_mm(b.ilk_delme_mm):.3f} mm"
              f" (bandın küçük ucu; testten sonra dış çap büyütülerek açılır)")
        print(f"  >> Tasarım modeli: aralık {b.d_tasarim_mm:.3f} mm, dış çap {b.D_dis_tasarim_mm:.3f} mm | "
              f"Dh = {2*b.d_tasarim_mm:.3f} mm | L/Dh = {b.L_d:.1f} | Re = {b.Re_delik:.2e} | eksantriklik etkisi ±%{b.duzgunsuzluk_pct:.1f}")
    else:
        print(f"  >> İLK DELME ÖNERİSİ: {b.ilk_delme_mm:.2f} mm (bandın küçük ucu; soğuk akış testinden sonra rayba ile büyütün)")
        print(f"  >> Tasarım modeli çapı: {b.d_tasarim_mm:.3f} mm | L/d = {b.L_d:.2f} | Re_delik = {b.Re_delik:.2e}")
    print(f"  Jet hızı bandı (momentum oranı için): {fl.v_jet_sivi:.0f} m/s (flaşsız sıvı) .. {fl.v_jet_hem:.0f} m/s (tam flaş)"
          f" -> toplam N2O momentum akısı {op.m_dot_hedef*fl.v_jet_sivi:.0f} .. {op.m_dot_hedef*fl.v_jet_hem:.0f} N")

    print(f"\n[4] BESLEME HATTI")
    print(_cizgi())
    print(f"  Tank {st.P_pa/1e5:.2f} -> tank çıkışı {fd.P_vana_oncesi_pa/1e5:.2f} -> vana sonrası {fd.P_vana_sonrasi_pa/1e5:.2f}"
          f" -> manifold {fd.P1_efektif_pa/1e5:.2f} bar (toplam kayıp {fd.dP_toplam_pa/1e5:.2f} bar)")
    print(f"  Kayıplar [bar]: tank çıkışı {fd.dP_tank_cikis_pa/1e5:.2f} | vana {fd.dP_vana_pa/1e5:.2f} | boru {fd.dP_boru_pa/1e5:.2f}"
          f" | fitting {fd.dP_fitting_pa/1e5:.2f} | manifold {fd.dP_manifold_pa/1e5:.2f}")
    print(f"  Hat hızı {fd.v_hat_m_s:.2f} m/s (Re {fd.Re_hat:.2e}, f {fd.f_darcy:.4f}) | galeri hızı {fd.v_manifold_m_s:.2f} m/s"
          f" | hat/orifis alan oranı {fd.alan_orani_hat_orifis:.1f}")
    print(f"  Vana boğulma sınırı dP_kr = {fd.dP_vana_bogulma_pa/1e5:.2f} bar -> {'KAVİTASYON' if fd.vana_kavitasyon else 'OK'}")
    print(f"  Manifold alt-soğutma = {fd.alt_sogutma_manifold_bar:.2f} bar | MAKSİMUM TANK SICAKLIĞI (manifoldda kaynamasız):"
          f" {fd.T_tank_max_K-273.15:.1f} °C (şu an {op.T_tank_K-273.15:.1f} °C)")
    for w in fd.uyarilar:
        print(f"    [!] {w}")

    print(f"\n[5] AYNA TERMAL ({t.mat_etiket}, N2O ıslak alan {t.A_ayna_m2*1e4:.1f} cm2 = Ø{th.D_ayna_mm:.1f} - Ø{th.D_merkez_mm:.1f} mm)")
    print(_cizgi())
    print(f"  h_N2O [W/m2K]: galeri {t.h_n2o_manifold:.0f} | delik girişi {t.h_n2o_giris:.0f} | delik {t.h_n2o_delik:.0f}"
          f" | aynaya indirgenmiş {t.h_n2o_efektif:.0f}")
    print(f"  Yanma sonu ({th.t_yanma_s:.1f} s, nominal): gaz yüzü {t.Tw_gaz_tb_K-273.15:.0f} °C | N2O yüzü {t.Tw_n2o_tb_K-273.15:.1f} °C"
          f" | Q = {t.Q_tb_W/1e3:.2f} kW | bulk ısınma +{t.dT_bulk_nom_K:.2f} K (üst sınır +{t.dT_bulk_max_K:.2f} K)")
    print(f"  Kararlı hal referansı: gaz yüzü {t.Tw_gaz_kararli_K-273.15:.0f} °C | N2O yüzü {t.Tw_n2o_kararli_K-273.15:.1f} °C"
          f" {'' if t.fsolve_ok else '[fsolve yakınsamadı]'}")
    print(f"  Galeri T_sat = {t.T_sat_manifold_K-273.15:.1f} °C | Davis-Anderson T_ONB = {t.T_onb_K-273.15:.1f} °C")
    print(f"  Kapanma sonrası: depolanan ısı {t.E_depolanan_nom_J/1e3:.1f} kJ (nominal) / {t.E_depolanan_max_J/1e3:.1f} kJ (en kötü)"
          f" | hapsolan N2O {t.m_hapsolan_kg*1e3:.1f} g, kaynatmak için {t.E_hapsolan_kaynama_J/1e3:.1f} kJ")
    print(f"    {th.t_kapanma_sonrasi_s:.0f} s içinde N2O yüzü max {t.Tw_n2o_soak_max_K-273.15:.0f} °C (gaz yüzü adyabatik, en kötü zarf)"
          f" | N2O kaynadıktan sonra ayna denge sıcaklığı: nominal {t.T_denge_kuru_nom_K-273.15:.0f} °C,"
          f" en kötü {t.T_denge_kuru_K-273.15:.0f} °C")
    print(f"    {'HAPSOLAN N2O TAMAMEN BUHARLAŞIR' if t.hapsolan_tamamen_kaynar else 'Hapsolan N2O ısıyı emebilir'}"
          f" (oda duvarlarından ışınım ihmal edildi)")
    print(f"  DURUM: {t.durum_ozeti}")
    print(f"  {'T_gaz [K]':>9} | {'h_gaz':>6} | {'Gaz yüzü °C':>11} | {'N2O yüzü °C':>11} | {'Q kW':>6} | dT_bulk K   (t = {th.t_yanma_s:.1f} s)")
    for Tg, hg, twg, twn, q, dtb in t.grid:
        print(f"  {Tg:>9.0f} | {hg:>6.0f} | {twg:>11.0f} | {twn:>11.1f} | {q:>6.2f} | +{dtb:.2f}")

    print(f"\n[6] İLERİ YÖNLÜ {'ARALIK' if orif.annulus else 'MATKAP'} TABLOSU (P2 = {P2b:.1f} bar sabit, hat kaybı bağlaşık, debiler kg/s)")
    print(_cizgi())
    kol = ('spi',) + BANT_MODELLERI
    print("  " + f"{('g mm' if orif.annulus else 'd mm'):>5} | " + " | ".join(f"{k:>9}" for k in kol) + f" | {'±tol(tas.)':>11} | {'dlnm/dlnP2':>10} | Not")
    for r in matkap:
        print("  " + f"{r.d_mm:>5.2f} | " + " | ".join(f"{r.mdot[k]:>9.3f}" for k in kol)
              + f" | {r.mdot_tol_eksi:.3f}-{r.mdot_tol_arti:.3f} | {r.dlnm_dlnP2:>10.3f} | {r.not_str}")
    print("  dlnm/dlnP2 ~ 0: debi oda basıncından bağımsız (boğulmuş/kavitasyonlu), oda dalgalanmalarına karşı iyi.")

    print(f"\n[7] DUYARLILIK VE MONTE CARLO (tasarım modeli: {tm})")
    print(_cizgi())
    print(f"  d(lnG)/d(lnP1) = {uq.dlnG_dlnP1:.2f} | dG/dT1 = %{uq.dG_dT1_pct_K:+.2f} / K")
    print(f"  Gereken {bad}, yalnız donanım/girdi belirsizliği : P10 {uq.d_hw_P10:.3f} | P50 {uq.d_hw_P50:.3f} | P90 {uq.d_hw_P90:.3f} mm")
    print(f"  Gereken {bad}, + model seçimi (bant, eşit ağırlık): P10 {uq.d_karma_P10:.3f} | P50 {uq.d_karma_P50:.3f} | P90 {uq.d_karma_P90:.3f} mm")
    print(f"  İlk {'işleme' if orif.annulus else 'delme'} {uq.ilk_delme_mm:.2f} mm ile beklenen debi: donanım P10/P50/P90 = {uq.mdot_ilk_hw_P10:.3f}/"
          f"{uq.mdot_ilk_hw_P50:.3f}/{uq.mdot_ilk_hw_P90:.3f} | +model = {uq.mdot_ilk_karma_P10:.3f}/"
          f"{uq.mdot_ilk_karma_P50:.3f}/{uq.mdot_ilk_karma_P90:.3f} kg/s (hedef {op.m_dot_hedef:.3f})")
    print(f"  Tornado (gereken {'aralığı' if orif.annulus else 'çapı'} en çok oynatan etkenler):")
    for i, (ad, (lo, hi, sp)) in enumerate(uq.tornado.items(), 1):
        print(f"    {i}. {ad:<52}: [{lo:.3f} .. {hi:.3f}] mm  fark {sp:.3f}  {'█' * int(np.clip(round(sp * 60), 0, 32))}")
    for nt in uq.notlar:
        print(f"    * {nt}")

    print(f"\n[8] KALİBRASYON")
    print(_cizgi())
    if kal_ozet:
        for k, (km, ks, nn) in kal_ozet.items():
            print(f"  {MODEL_ADLARI[k]:<46}: k = {km:.3f} (std {ks:.3f}, n = {nn})")
    else:
        print("  Test verisi yok. Su testiyle Cd_tekfaz'ı, N2O testiyle model çarpanlarını belirleyin")
        print("  (KALIBRASYON_TESTLERI listesine ekleyin; tasarım koşuluna yakın alt-soğutma ve P2 tercih edin).")
    print(_cizgi('=') + "\n")


# %% [14] ÇALIŞTIRMA  (TÜM GİRDİLER BURADA)
# Kendi soğuk akış testleriniz. Orifis testi:
#   KalibrasyonTesti(P1_bar=55.0, T1_K=290.0, P2_bar=1.013, d_mm=2.20, N_orifis=8, mdot_olculen=0.95)
# Annulus testi (d_mm = aralık, N_orifis = 1, D_ic_mm = iç çap):
#   KalibrasyonTesti(P1_bar=55.0, T1_K=290.0, P2_bar=1.013, d_mm=0.45, N_orifis=1, mdot_olculen=0.95, D_ic_mm=22.0)
KALIBRASYON_TESTLERI: List[KalibrasyonTesti] = []

# Hangi senaryolar çalışsın: 'orifis', 'annulus' veya ikisi birden
SENARYOLAR = ['orifis', 'annulus']

if __name__ == "__main__":
    # ---- Ortak işletme girdileri -------------------------------------------------
    op = OperatingConfig(
        m_dot_rpa=1.09608, eta_atomizasyon=0.90,
        P_tank_pa=62.0e5, T_tank_K=293.10, P2_pa=27.0e5,
        manuel_x1=None,
        exponent_mode='real_ns', vap_ref_mode='T1_sat', slip_mode='zivi',
        tasarim_modeli='dual_spc',
    )
    feed = FeedSystemConfig(
        D_hat_mm=15.7, L_hat_m=1.0, eps_boru_mm=0.015,
        K_tank_cikis=0.5, Cv_ana_vana=5.5, FL_vana=0.60, K_fitting_toplam=0.8,
        K_manifold_giris=1.5, D_manifold_ic_mm=60.0, H_manifold_ic_mm=10.0,
    )
    th = ThermalConfig(
        T_gaz_eff_low_K=1400.0, T_gaz_eff_nom_K=2050.0, T_gaz_eff_high_K=2453.0,
        h_gaz_low=500.0, h_gaz_nominal=950.0, h_gaz_high=1600.0,
        D_ayna_mm=83.4, D_merkez_mm=20.0,            # <- gerçek ölçülerle değiştir
        malzeme='Inconel_718', t_yanma_s=5.0,
        t_kapanma_sonrasi_s=30.0, h_n2o_durgun=300.0,
        V_hapsolan_cm3=25.0,                         # <- vana-ayna arası gerçek hacim
        f_mix_nominal=0.35,
        T_wall_caution_K=473.15, T_wall_hazard_K=573.15,
    )

    # ---- Senaryo 1: ORİFİS (N adet silindirik delik) ---------------------------
    orif_orifis = OrificeConfig(
        geometri='orifis',
        Cd_tekfaz=0.60, N_orifis=8, L_orifis_mm=8.0,
        inlet_edge_type='sharp', Cc_override=None, drill_tolerance_mm=0.02,
    )
    matkaplar = [round(x, 2) for x in np.arange(2.00, 2.951, 0.05)]       # taranacak delik çapları [mm]

    # ---- Senaryo 2: ANNULUS (tek halka aralık) ----------------------------------
    orif_annulus = OrificeConfig(
        geometri='annulus',
        D_ic_mm=22.0,                  # <- halkanın iç çapı [mm] (merkez gövde dış çapı)
        L_orifis_mm=8.0,               # <- halka kanal boyu [mm]
        Cd_tekfaz=0.75,                # <- KAVİTASYONSUZ tek faz Cd (keskin giriş, L/Dh~6.5 için ~0.75-0.8); su testiyle ölçün
        inlet_edge_type='sharp', Cc_override=None,
        drill_tolerance_mm=0.02,       # <- aralık toleransı [±mm]
        eksantriklik_mm=0.02,          # <- iç/dış merkez kaçıklığı [mm]
    )
    araliklar = [round(x, 2) for x in np.arange(0.35, 0.851, 0.05)]       # taranacak aralıklar [mm]

    # ---- Çalıştırma (bu satırların altına dokunmayın) ----------------------------
    vv = vv_makale_karsilastirmasi()
    senaryo_tanim = {'orifis': (orif_orifis, matkaplar), 'annulus': (orif_annulus, araliklar)}
    for i, ad in enumerate(SENARYOLAR):
        orif, boyutlar = senaryo_tanim[ad]
        kal_ozet = kalibrasyon_hesapla(KALIBRASYON_TESTLERI, op, orif) if KALIBRASYON_TESTLERI else {}
        kal = {k: v[0] for k, v in kal_ozet.items()}
        sol = baglasik_coz(op, orif, feed, th, kal)
        tablo = matkap_tablosu_coz(sol, op, orif, feed, kal, boyutlar)
        uq = belirsizlik_coz(sol, op, orif, kal)
        rapor_yazdir(op, orif, feed, th, vv, sol, tablo, uq, kal_ozet, vv_yazdir=(i == 0))