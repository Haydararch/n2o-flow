# -*- coding: utf-8 -*-
"""
===============================================================================
N2O ENJEKTÖR AKIŞ, BESLEME HATTI VE AYNA TERMAL HESABI  (V7.3: orifis + annulus + manifold)
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

V7.3 eklemeleri
  * ManifoldConfig (annulus modu): halka manifold iç/dış çap ve eksenel boy, giriş portları.
    Manifold kaybı port ve dönüş dinamik basınçlarından hesaplanır (eski K x hat dinamik basıncı
    yalnızca orifis modunda / manifold tanımsızken kullanılır).
  * manifold_coz: manifoldda hal durumu (alt-soğutma, faz, void, buhar hacmi), çevresel dağılım,
    port jeti ve annulus girişi kavitasyon öngörüleri, Wood ses hızı, hidrolik esneklik,
    kaba Helmholtz frekansı, buhar kabarcığı birikme riski.
  * sicaklik_taramasi: 0-40 °C tank sıcaklığı taraması (sabit aralıkta debi, O/F, gereken aralık).
    Tank basıncı sabit (He regülatörlü) kabul edilir.
  * senaryo_coz: S1 sıvı->sıvı, S2 sıvı->karışık, S3 karışık (x1 bilinen), S4 karışık (x1 taraması).
    S3/S4 için Leung omega yöntemi eklendi (omega_akisi).
  * Taşınım özellikleri önbelleği 0.1 bar / 0.1 K'e yuvarlanır (hız).

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
class ManifoldConfig:
    """
    Annulus modu için N2O manifoldu: halka oda (iç çap D_ic, dış çap D_dis, eksenel boy L).
    N2O, N_giris adet porttan (çap d_giris) girer, çevresel olarak iki yöne dağılır, eksenel
    ilerleyerek manifold tabanındaki annulus girişine döner.
    """
    D_ic_mm: float = 14.0               # Manifold iç çapı [mm] (GAZAP: pintle dış çapı)
    D_dis_mm: float = 44.0              # Manifold dış çapı [mm]
    L_mm: float = 8.0                   # Manifold eksenel boyu (derinlik) [mm]
    N_giris: int = 2                    # Giriş portu sayısı
    d_giris_mm: float = 10.0            # Giriş portu çapı [mm] (toplam port alanı ~ hat alanı olmalı)
    K_giris: float = 1.0                # Port çıkışı ani genişleme kaybı (port dinamik basıncıyla) [-]
    K_donus: float = 0.5                # Eksenel akıştan annulus girişine dönüş kaybı (eksenel din. basınçla) [-]
    sigma_i: float = 2.0                # Port jeti kavitasyon başlangıç sayısı (keskin kenar ~1.5-2.5) [-]

    def __post_init__(self):
        if self.D_dis_mm <= self.D_ic_mm:
            raise ValueError("Manifold dış çapı iç çapından büyük olmalı.")
        if self.L_mm <= 0 or self.d_giris_mm <= 0 or self.N_giris < 1:
            raise ValueError("Manifold boyu, port çapı > 0 ve port sayısı >= 1 olmalı.")

    @property
    def W_m(self) -> float:
        return (self.D_dis_mm - self.D_ic_mm) / 2.0 * 1e-3

    @property
    def V_m3(self) -> float:
        return np.pi / 4.0 * ((self.D_dis_mm * 1e-3) ** 2 - (self.D_ic_mm * 1e-3) ** 2) * self.L_mm * 1e-3

    @property
    def A_eksenel_m2(self) -> float:
        return np.pi / 4.0 * ((self.D_dis_mm * 1e-3) ** 2 - (self.D_ic_mm * 1e-3) ** 2)

    @property
    def A_cevresel_m2(self) -> float:
        return self.W_m * self.L_mm * 1e-3

    @property
    def Dh_cevresel_m(self) -> float:
        W, L = self.W_m, self.L_mm * 1e-3
        return 2.0 * W * L / (W + L)

    @property
    def A_port_m2(self) -> float:
        return self.N_giris * np.pi * (self.d_giris_mm * 1e-3) ** 2 / 4.0


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
    manifold: Optional[ManifoldConfig] = None   # V7.3: tanımlıysa manifold kaybı/hızı buradan hesaplanır


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
    mu_l, k_l, Pr_l, _ = _n2o_transport(round(T_liq_eval, 1), round(P_pa, -4))

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
    Dh_manifold_m: float = float('nan')  # V7.3: termal için manifold hidrolik çapı
    v_port_m_s: float = float('nan')
    v_eksenel_m_s: float = float('nan')
    dP_port_pa: float = 0.0
    dP_donus_pa: float = 0.0


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

    mf = feed.manifold
    if mf is not None:
        # V7.3: halka manifold. Çevresel hız: her port debisi iki yöne ayrılır.
        v_port = m_dot / (rho * mf.A_port_m2)
        v_eks = m_dot / (rho * mf.A_eksenel_m2)
        v_man = m_dot / (rho * 2.0 * mf.N_giris * mf.A_cevresel_m2)
        dP_port = mf.K_giris * 0.5 * rho * v_port ** 2
        dP_donus = mf.K_donus * 0.5 * rho * v_eks ** 2
        Dh_man = mf.Dh_cevresel_m
    else:
        A_man = np.pi * feed.D_manifold_ic_mm * 1e-3 * feed.H_manifold_ic_mm * 1e-3   # radyal akış alanı
        v_man = m_dot / (rho * A_man)
        v_port = v_eks = float('nan')
        dP_port = dP_donus = 0.0
        Dh_man = 2.0 * feed.H_manifold_ic_mm * 1e-3

    SG = rho / 999.0
    Q_m3h = m_dot / rho * 3600.0
    Kv = 0.865 * feed.Cv_ana_vana
    dP_tank = feed.K_tank_cikis * q
    dP_vana = SG * (Q_m3h / Kv) ** 2 * 1e5
    dP_boru = f * (feed.L_hat_m / D) * q
    dP_fit = feed.K_fitting_toplam * q
    dP_man = (dP_port + dP_donus) if mf is not None else feed.K_manifold_giris * q
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
        vana_kavitasyon=bool(vana_kav), T_tank_max_K=float(T_max), uyarilar=uy,
        Dh_manifold_m=float(Dh_man), v_port_m_s=float(v_port), v_eksenel_m_s=float(v_eks),
        dP_port_pa=float(dP_port), dP_donus_pa=float(dP_donus)
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
    Dh_man = fd.Dh_manifold_m if np.isfinite(fd.Dh_manifold_m) else 2.0 * feed.H_manifold_ic_mm * 1e-3
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


# %% [11b] MANİFOLD ANALİZİ (V7.3, annulus modu)
@dataclass
class ManifoldSonucu:
    V_cm3: float
    A_port_mm2: float
    A_cevresel_mm2: float               # tek yön çevresel kesit (W x L)
    A_eksenel_mm2: float
    A_annulus_mm2: float
    oran_cevresel: float                # 2*N*A_cevresel / A_annulus (dağılım için >~3 önerilir)
    oran_eksenel: float                 # A_eksenel / A_annulus
    v_port_m_s: float
    v_cevresel_m_s: float
    v_eksenel_m_s: float
    Re_cevresel: float
    dagilim_pct: float                  # çevresel dinamik basınç / enjektör ΔP [%] (<~5 önerilir)
    P_bar: float
    T_C: float
    Pv_bar: float                       # Pv(T) (iki fazda = P)
    Tsat_C: float                       # Tsat(P)
    alt_sogutma_bar: float
    alt_sogutma_K: float
    faz: str
    x: float
    alpha: float
    rho_mix: float
    m_n2o_g: float
    V_buhar_cm3: float
    kalma_suresi_ms: float
    sigma_port: float
    port_kavitasyon: bool
    annulus_K: float                    # Nurick K
    annulus_K_kr: float
    annulus_kavitasyon: bool
    annulus_flas: bool                  # Pv(T1) > P2: çıkışta flaş kaçınılmaz
    kappa_dyer: float
    tau_r_us: float                     # kanalda kalma süresi
    tau_b_us: float                     # Dyer kabarcık büyüme süresi
    a_ses_m_s: float                    # Wood ses hızı (iki fazda donmuş karışım)
    C_cm3_bar: float                    # hidrolik esneklik V/(rho a^2)
    f_helmholtz_Hz: float               # manifold + annulus kaba Helmholtz frekansı
    u_kabarcik_m_s: float               # Harmathy kabarcık yükselme hızı
    kabarcik_birikme: bool              # eksenel hız < kabarcık hızı (dik motor, enjektör üstte)
    uyarilar: List[str]


def manifold_coz(m_dot: float, st: FluidState, mf: ManifoldConfig, orif: OrificeConfig, d_mm: float,
                 fl: AkisSonucu, P2_pa: float) -> ManifoldSonucu:
    """st: manifold (annulus girişi) durumu. d_mm: annulus aralığı. fl: aynı durum için akış sonucu."""
    uy: List[str] = []
    Pm = st.P_pa
    rho = st.rho_mix
    A_ann = float(orif.alan_m2(d_mm))
    Tsat = st.Tsat_P_K
    iki_faz = st.x > 0.0 or st.alt_sogutma_pa <= 100.0
    T_liq = Tsat - 1e-3 if iki_faz else st.T_K

    v_port = m_dot / (rho * mf.A_port_m2)
    v_cev = m_dot / (rho * 2.0 * mf.N_giris * mf.A_cevresel_m2)
    v_eks = m_dot / (rho * mf.A_eksenel_m2)
    Re_cev = rho * v_cev * mf.Dh_cevresel_m / st.mu_liq
    dP_inj = max(Pm - P2_pa, 1.0)
    dag = 100.0 * 0.5 * rho * v_cev ** 2 / dP_inj
    oran_c = 2.0 * mf.N_giris * mf.A_cevresel_m2 / A_ann
    oran_e = mf.A_eksenel_m2 / A_ann

    alt_K = (Tsat - st.T_K) if not iki_faz else 0.0
    V = mf.V_m3
    m_n2o = rho * V
    V_buhar = st.alpha * V
    tau_kal = V * rho / m_dot

    sig_p = max(Pm - st.Pv_pa, 0.0) / (0.5 * rho * v_port ** 2)
    port_kav = sig_p < mf.sigma_i
    flas = st.Pv_pa > P2_pa
    tau_b = float(np.sqrt(1.5 * st.rho_liq / (st.Pv_pa - P2_pa))) if flas else float('inf')

    a_l = CP.PropsSI('A', 'P', Pm, 'T', T_liq, FLUID)
    if st.alpha > 0.0:
        rl = CP.PropsSI('D', 'P', Pm, 'Q', 0, FLUID)
        rv = CP.PropsSI('D', 'P', Pm, 'Q', 1, FLUID)
        av = CP.PropsSI('A', 'P', Pm, 'T', Tsat + 1e-3, FLUID)
        a_ = st.alpha
        a_mix = float(1.0 / np.sqrt((a_ * rv + (1 - a_) * rl) * (a_ / (rv * av ** 2) + (1 - a_) / (rl * a_l ** 2))))
    else:
        a_mix = float(a_l)
    C = V / (rho * a_mix ** 2)
    L_eff = orif.L_orifis_mm * 1e-3 + orif.hidrolik_cap_m(d_mm)
    f_h = a_mix / (2.0 * np.pi) * np.sqrt(A_ann / (V * L_eff))

    sig_t = _n2o_transport(round(Tsat - 1e-3, 1), round(min(Pm, P_CRIT - 5e4), -4))[3]
    rls = CP.PropsSI('D', 'P', Pm, 'Q', 0, FLUID)
    rvs = CP.PropsSI('D', 'P', Pm, 'Q', 1, FLUID)
    u_b = 1.53 * (9.81 * sig_t * (rls - rvs) / rls ** 2) ** 0.25
    birikme = v_eks < u_b

    # uyarılar
    if orif.annulus:
        if orif.D_ic_mm < mf.D_ic_mm - 1e-6 or orif.D_dis_mm(d_mm) > mf.D_dis_mm + 1e-6:
            uy.append(f"GEOMETRİ: annulus (Ø{orif.D_ic_mm:.2f}-Ø{orif.D_dis_mm(d_mm):.2f}) manifold tabanının "
                      f"(Ø{mf.D_ic_mm:.1f}-Ø{mf.D_dis_mm:.1f}) dışına taşıyor.")
    if iki_faz:
        uy.append(f"MANİFOLDDA İKİ FAZ: x = %{st.x*100:.1f}, void = %{st.alpha*100:.0f}, buhar hacmi {V_buhar*1e6:.1f} cm³. "
                  f"Ses hızı {a_mix:.0f} m/s'ye, Helmholtz frekansı {f_h:.0f} Hz'e düşer (chugging riski).")
    elif st.alt_sogutma_pa < 3e5:
        uy.append(f"Manifold alt-soğutma marjı düşük: {st.alt_sogutma_pa/1e5:.2f} bar ({alt_K:.1f} K).")
    if dag > 5.0:
        uy.append(f"DAĞILIM: çevresel dinamik basınç enjektör ΔP'sinin %{dag:.1f}'i (> %5). Port sayısını artırın "
                  f"veya manifold kesitini büyütün.")
    if oran_c < 3.0:
        uy.append(f"Çevresel kesit / annulus alanı = {oran_c:.1f} (< 3): çevresel debi dağılımı sınırda.")
    if port_kav:
        uy.append(f"PORT KAVİTASYONU: σ = {sig_p:.2f} < σ_i = {mf.sigma_i:.1f} (port hızı {v_port:.1f} m/s). "
                  f"Port çapını/sayısını artırın.")
    if birikme:
        uy.append(f"Eksenel hız {v_eks:.2f} m/s < kabarcık yükselme hızı {u_b:.2f} m/s: buhar oluşursa (açılış, "
                  f"ısınma, iki fazlı besleme) manifold üstünde toplanabilir. Portları üst (kapak) tarafa koyun.")
    return ManifoldSonucu(
        V_cm3=V * 1e6, A_port_mm2=mf.A_port_m2 * 1e6, A_cevresel_mm2=mf.A_cevresel_m2 * 1e6,
        A_eksenel_mm2=mf.A_eksenel_m2 * 1e6, A_annulus_mm2=A_ann * 1e6, oran_cevresel=float(oran_c),
        oran_eksenel=float(oran_e), v_port_m_s=float(v_port), v_cevresel_m_s=float(v_cev), v_eksenel_m_s=float(v_eks),
        Re_cevresel=float(Re_cev), dagilim_pct=float(dag), P_bar=Pm / 1e5, T_C=T_liq - 273.15 if iki_faz else st.T_K - 273.15,
        Pv_bar=st.Pv_pa / 1e5, Tsat_C=Tsat - 273.15, alt_sogutma_bar=st.alt_sogutma_pa / 1e5, alt_sogutma_K=float(alt_K),
        faz=st.faz, x=st.x, alpha=st.alpha, rho_mix=float(rho), m_n2o_g=float(m_n2o * 1e3), V_buhar_cm3=float(V_buhar * 1e6),
        kalma_suresi_ms=float(tau_kal * 1e3), sigma_port=float(sig_p), port_kavitasyon=bool(port_kav),
        annulus_K=fl.K_nurick, annulus_K_kr=fl.K_kritik, annulus_kavitasyon=fl.kavitasyonlu, annulus_flas=bool(flas),
        kappa_dyer=fl.kappa_dyer, tau_r_us=fl.tau_r_s * 1e6, tau_b_us=tau_b * 1e6, a_ses_m_s=a_mix,
        C_cm3_bar=float(C * 1e6 * 1e5), f_helmholtz_Hz=float(f_h), u_kabarcik_m_s=float(u_b),
        kabarcik_birikme=bool(birikme), uyarilar=uy)


# %% [11c] LEUNG OMEGA YÖNTEMİ (doymuş / iki fazlı giriş)
def omega_akisi(P0_pa: float, x0: float, P2_pa: float, Cd: float = 1.0) -> Tuple[float, float, float, bool]:
    """
    Leung (1986) omega yöntemi, P0'da doymuş giriş (kuruluk x0). Döndürür: (G [kg/m2s], omega, eta_kr, boğulma).
    """
    vl = 1.0 / CP.PropsSI('D', 'P', P0_pa, 'Q', 0, FLUID)
    vv = 1.0 / CP.PropsSI('D', 'P', P0_pa, 'Q', 1, FLUID)
    hlg = CP.PropsSI('H', 'P', P0_pa, 'Q', 1, FLUID) - CP.PropsSI('H', 'P', P0_pa, 'Q', 0, FLUID)
    T0 = CP.PropsSI('T', 'P', P0_pa, 'Q', 0, FLUID)
    cpl = CP.PropsSI('C', 'P', P0_pa, 'Q', 0, FLUID)
    v0 = x0 * vv + (1.0 - x0) * vl
    w = x0 * (vv - vl) / v0 + cpl * T0 * P0_pa / v0 * ((vv - vl) / hlg) ** 2

    def f(e):
        return e ** 2 + (w ** 2 - 2.0 * w) * (1.0 - e) ** 2 + 2.0 * w ** 2 * np.log(e) + 2.0 * w ** 2 * (1.0 - e)
    e_kr = root_scalar(f, bracket=[1e-6, 1.0 - 1e-9], method='brentq').root
    eta = max(P2_pa / P0_pa, e_kr)
    G = np.sqrt(P0_pa / v0) * np.sqrt(max(-2.0 * (w * np.log(eta) + (w - 1.0) * (1.0 - eta)), 0.0)) / (w * (1.0 / eta - 1.0) + 1.0)
    return float(Cd * G), float(w), float(e_kr), bool(P2_pa / P0_pa <= e_kr)


# %% [11d] SABİT ARALIKTA DEBİ, SICAKLIK TARAMASI VE SENARYOLAR (V7.3)
SENARYO_MODEL_ADLARI: Dict[str, str] = {**MODEL_ADLARI, 'omega': 'Leung ω (doymuş/iki fazlı giriş)'}


def sabit_aralik_debi(op: OperatingConfig, orif: OrificeConfig, feed: FeedSystemConfig,
                      kal: Optional[Dict[str, float]], d_mm: float, st_tank: FluidState, T_in_K: float,
                      modeller: Tuple[str, ...], tol: float = 2e-4) -> Tuple[Dict[str, float], Dict[str, tuple]]:
    """Sabit boyutta (aralık/çap) her model için debi; hat kaybı debiyle bağlaşık. (debiler, {model: (fd, st, fl)})"""
    A = float(orif.alan_m2(d_mm))
    out: Dict[str, float] = {}
    ara: Dict[str, tuple] = {}
    for mk in modeller:
        md = op.m_dot_hedef
        try:
            for j in range(30):
                fd = besleme_sistemi_coz(md, st_tank, feed, A * 1e6)
                if fd.P1_efektif_pa <= op.P2_pa + 1e5:
                    raise ValueError("P1 ~ P2")
                st = durum_op(fd.P1_efektif_pa, T_in_K, op)
                fl = akis_op(st, op, orif, kal)
                md_y = fl.G_kal[mk] * A
                if abs(md_y - md) < tol:
                    md = md_y
                    break
                md = md_y if j < 3 else 0.5 * (md + md_y)
            out[mk] = float(md)
            ara[mk] = (fd, st, fl)
        except ValueError:
            out[mk] = float('nan')
    return out, ara


def _OF(md: float, op: OperatingConfig, OF_rpa: float) -> float:
    """Yakıt debisi sabit; hedef N2O debisinde O/F = RPA O/F kabulü."""
    return OF_rpa * md / op.m_dot_hedef


@dataclass
class SicaklikSatiri:
    T_C: float
    durum: str                          # 'OK' veya neden geçersiz olduğu
    Pv_bar: float = float('nan')
    alt_sogutma_tank_bar: float = float('nan')
    P_man_bar: float = float('nan')
    alt_sogutma_man_bar: float = float('nan')
    alt_sogutma_man_K: float = float('nan')
    rho_l: float = float('nan')
    mdot: Dict[str, float] = field(default_factory=dict)      # sabit aralıkta
    OF_tasarim: float = float('nan')
    OF_min: float = float('nan')
    OF_max: float = float('nan')
    sapma_pct: float = float('nan')     # tasarım modeli, hedefe göre
    d_gerekli: Dict[str, float] = field(default_factory=dict)
    d_bant_min: float = float('nan')
    d_bant_max: float = float('nan')
    K_nurick: float = float('nan')
    kavitasyonlu: bool = False
    kappa: float = float('nan')
    hem_bogulmus: bool = False
    x2: float = float('nan')
    v_jet_sivi: float = float('nan')
    sigma_port: float = float('nan')
    dag_pct: float = float('nan')


def sicaklik_taramasi(op: OperatingConfig, orif: OrificeConfig, feed: FeedSystemConfig,
                      kal: Optional[Dict[str, float]], d_mm: float, T_liste_C, OF_rpa: float = 4.0) -> List[SicaklikSatiri]:
    """
    Tank sıvı sıcaklığı taraması. Tank basıncı sabit (He regülatörlü). Termal bulk ısınma ihmal (adyabatik).
    d_mm: sabit tutulan boyut (aralık/çap). Her sıcaklıkta ayrıca hedef debi için gereken boyut verilir.
    """
    kol = ('spi',) + BANT_MODELLERI
    tm = op.tasarim_modeli
    satirlar: List[SicaklikSatiri] = []
    for T_C in T_liste_C:
        T_K = float(T_C) + 273.15
        if T_K >= T_CRIT - 0.05:
            satirlar.append(SicaklikSatiri(float(T_C), f"KRİTİK ÜSTÜ (Tkr = {T_CRIT-273.15:.1f} °C): sıvı N2O yok"))
            continue
        opT = replace(op, T_tank_K=T_K)
        try:
            st_tank = durum_op(op.P_tank_pa, T_K, opT)
        except ValueError:
            Pv = CP.PropsSI('P', 'T', T_K, 'Q', 0, FLUID)
            satirlar.append(SicaklikSatiri(float(T_C), f"TANKTA KAYNAMA: Pv = {Pv/1e5:.1f} bar > tank {op.P_tank_pa/1e5:.1f} bar",
                                           Pv_bar=Pv / 1e5))
            continue
        try:
            fd = besleme_sistemi_coz(op.m_dot_hedef, st_tank, feed, float(orif.alan_m2(d_mm)) * 1e6)
            st = durum_op(fd.P1_efektif_pa, T_K, opT)
            fl = akis_op(st, opT, orif, kal)
            boy = orifis_boyutlandir(op.m_dot_hedef, fl, st, opT, orif)
        except ValueError:
            Pv = CP.PropsSI('P', 'T', T_K, 'Q', 0, FLUID)
            fd0 = besleme_sistemi_coz(op.m_dot_hedef, st_tank, feed, float(orif.alan_m2(d_mm)) * 1e6)
            satirlar.append(SicaklikSatiri(float(T_C), f"HATTA KAYNAMA: manifold {fd0.P1_efektif_pa/1e5:.1f} bar < Pv {Pv/1e5:.1f} bar",
                                           Pv_bar=Pv / 1e5, alt_sogutma_tank_bar=st_tank.alt_sogutma_pa / 1e5,
                                           P_man_bar=fd0.P1_efektif_pa / 1e5))
            continue
        md, _ = sabit_aralik_debi(opT, orif, feed, kal, d_mm, st_tank, T_K, kol)
        bant = [md[k] for k in BANT_MODELLERI if np.isfinite(md.get(k, np.nan)) and not (k == 'nurick' and fl.K_nurick < 0.05)]
        mt = md.get(tm, float('nan'))
        if tm not in md:
            mt = sabit_aralik_debi(opT, orif, feed, kal, d_mm, st_tank, T_K, (tm,))[0][tm]
        a_s = st.alt_sogutma_pa / 1e5
        durum = 'OK' if a_s >= 3.0 else ('DÜŞÜK MARJ (alt-soğutma < 3 bar)' if a_s >= 0.5 else 'DOYMUŞ SINIR (manifoldda kaynama eşiği)')
        sat = SicaklikSatiri(
            T_C=float(T_C), durum=durum, Pv_bar=st_tank.Pv_pa / 1e5, alt_sogutma_tank_bar=st_tank.alt_sogutma_pa / 1e5,
            P_man_bar=fd.P1_efektif_pa / 1e5, alt_sogutma_man_bar=st.alt_sogutma_pa / 1e5,
            alt_sogutma_man_K=st.Tsat_P_K - T_K, rho_l=st.rho_liq, mdot=md,
            OF_tasarim=_OF(mt, op, OF_rpa), OF_min=_OF(min(bant), op, OF_rpa) if bant else float('nan'),
            OF_max=_OF(max(bant), op, OF_rpa) if bant else float('nan'),
            sapma_pct=100.0 * (mt / op.m_dot_hedef - 1.0), d_gerekli=boy.d_mm, d_bant_min=boy.bant_min_mm,
            d_bant_max=boy.bant_max_mm, K_nurick=fl.K_nurick, kavitasyonlu=fl.kavitasyonlu, kappa=fl.kappa_dyer,
            hem_bogulmus=fl.hem_bogulmus, x2=fl.x_2, v_jet_sivi=fl.v_jet_sivi)
        if feed.manifold is not None and orif.annulus:
            ms = manifold_coz(op.m_dot_hedef, st, feed.manifold, orif, d_mm, fl, op.P2_pa)
            sat.sigma_port, sat.dag_pct = ms.sigma_port, ms.dagilim_pct
        satirlar.append(sat)
    return satirlar


@dataclass
class SenaryoSonucu:
    kod: str
    ad: str
    giris: str
    ana_model: str
    mdot: Dict[str, float]              # sabit aralıkta
    d_gerekli: Dict[str, float]         # hedef debi için
    OF: Dict[str, float]
    mdot_min: float
    mdot_max: float
    OF_min: float
    OF_max: float
    kabul: str                          # KABUL / SINIRDA / RED
    man: Optional[ManifoldSonucu]
    tarama: List[dict]                  # S4: x1 taraması
    notlar: List[str]


def _kabul(OF: Dict[str, float], ana: str, OF_rpa: float, bant_pct: float) -> str:
    """Karar ana modele göre verilir; model bandının taşması ayrıca belirtilir."""
    if ana not in OF or not np.isfinite(OF[ana]):
        return "HAT KAYNAR (ana model debisinde manifold Pv altına düşer)"
    ic = {k: abs(v / OF_rpa - 1.0) * 100.0 <= bant_pct for k, v in OF.items() if np.isfinite(v)}
    if not ic[ana]:
        return f"RED (ana model O/F {OF[ana]:.2f}, bant ±%{bant_pct:.0f} dışı)"
    if all(ic.values()) and len(ic) == len(OF):
        return "KABUL"
    return "KABUL (ana model) / model bandı ±%{:.0f}'u aşıyor".format(bant_pct)


def senaryo_coz(op: OperatingConfig, orif: OrificeConfig, feed: FeedSystemConfig, kal: Optional[Dict[str, float]],
                sol: 'Cozum', d_mm: float, OF_rpa: float = 4.0, OF_bant_pct: float = 10.0, x1_bilinen: float = 0.05,
                x1_liste: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)) -> List[SenaryoSonucu]:
    """
    S1 sıvı->sıvı, S2 sıvı->karışık: nominal sıvı giriş, hat kaybı bağlaşık, sabit aralık.
    S3/S4 karışık giriş: nominal manifold basıncı P1'de DOYMUŞ karışım (T1 = Tsat(P1)), kuruluk x1.
    He basınçlandırmalı sistemde bu durum ancak N2O hatta ısınırsa/basınç düşerse oluşur.
    """
    A = float(orif.alan_m2(d_mm))
    m_h = op.m_dot_hedef
    T_in = sol.st_in.T_K
    P1 = sol.fd.P1_efektif_pa
    Cd = orif.Cd_tekfaz
    out: List[SenaryoSonucu] = []

    def _paket(kod, ad, giris, ana, md, dg, man, tarama, notlar):
        OF = {k: _OF(v, op, OF_rpa) for k, v in md.items()}
        v = [x for x in md.values() if np.isfinite(x)]
        o = [x for x in OF.values() if np.isfinite(x)]
        return SenaryoSonucu(kod, ad, giris, ana, md, dg, OF, min(v) if v else np.nan, max(v) if v else np.nan,
                             min(o) if o else np.nan, max(o) if o else np.nan, _kabul(OF, ana, OF_rpa, OF_bant_pct),
                             man, tarama, notlar)

    # ---- S1 / S2: sıvı giriş
    mod12 = ('spi', 'spc_liq', 'dyer', 'dyer_comp', 'hem_c', 'fml', 'dual_spc', 'nurick')
    md_all, ara = sabit_aralik_debi(op, orif, feed, kal, d_mm, sol.st_tank, T_in, mod12)
    fl0, st0 = sol.fl, sol.st_in
    dg_all = {k: boyut_hesapla_mm(m_h, fl0.G_kal[k], orif) for k in mod12}
    man0 = None
    if feed.manifold is not None and orif.annulus:
        man0 = manifold_coz(m_h, st0, feed.manifold, orif, d_mm, fl0, op.P2_pa)
    giris12 = (f"Sıvı, manifold {P1/1e5:.2f} bar, {T_in-273.15:.1f} °C, alt-soğutma {st0.alt_sogutma_pa/1e5:.2f} bar")
    n1: List[str] = []
    if st0.Pv_pa <= op.P2_pa:
        n1.append(f"Pv(T1) = {st0.Pv_pa/1e5:.2f} bar ≤ P2 = {op.P2_pa/1e5:.1f} bar: çıkışta da sıvı, S1 fiziksel olarak geçerli.")
    else:
        tb = np.sqrt(1.5 * st0.rho_liq / (st0.Pv_pa - op.P2_pa))
        n1.append(f"Pv(T1) = {st0.Pv_pa/1e5:.2f} bar > P2 = {op.P2_pa/1e5:.1f} bar: çıkışta flaş kaçınılmaz. S1 yalnızca "
                  f"metastabil üst sınır. κ = {fl0.kappa_dyer:.2f}, τ_r = {fl0.tau_r_s*1e6:.0f} µs, τ_b = {tb*1e6:.0f} µs "
                  f"(κ ≫ 1 ise S1'e yakın, κ ≈ 1 ise S2).")
    n1.append("SPI/SPC kanal içinde buharlaşma olmadığını varsayar: debinin fiziksel üst sınırı.")
    kay = [k for k in ('spi', 'spc_liq', 'dyer') if not np.isfinite(md_all[k])]
    if kay:
        n1.append(f"{', '.join(kay)}: bu aralıkta bu modelin vereceği debide hat kayıpları manifoldu Pv'nin altına "
                  f"indiriyor (hatta kaynama). Yani akış besleme tarafından sınırlanır; S1 debisi pratikte erişilemez.")
    s1 = ('spi', 'spc_liq', 'dyer')
    out.append(_paket('S1', 'Sıvı giriyor, sıvı çıkıyor', giris12, 'spc_liq', {k: md_all[k] for k in s1},
                      {k: dg_all[k] for k in s1}, man0, [], n1))
    s2 = ('dyer_comp', 'hem_c', 'fml', 'dual_spc') + (('nurick',) if fl0.K_nurick >= 0.05 else ())
    ana2 = op.tasarim_modeli if op.tasarim_modeli in s2 else 'dual_spc'
    n2 = [f"Kanal içi buharlaşma: HEMc alt sınır (tam denge), ana model {ana2}. HEM boğulma basıncı "
          f"{fl0.P_hem_bogulma_pa/1e5:.2f} bar → {'boğulmuş' if fl0.hem_bogulmus else 'boğulmamış'}.",
          f"Annulus girişi Nurick K = {fl0.K_nurick:.3f}, K_kr = {fl0.K_kritik:.2f} → "
          f"{'KAVİTASYONLU' if fl0.kavitasyonlu else 'kavitasyonsuz'}. Çıkış kuruluğu %{fl0.x_2*100:.1f}, void %{fl0.alpha_2*100:.0f}."]
    out.append(_paket('S2', 'Sıvı giriyor, karışık çıkıyor', giris12, ana2, {k: md_all[k] for k in s2},
                      {k: dg_all[k] for k in s2}, man0, [], n2))

    # ---- S3 / S4: doymuş karışım girişi P1'de
    def _iki_faz(x1):
        st = termodinamik_durum_coz(P1, T_in, manuel_x=x1, exponent_mode=op.exponent_mode,
                                    vap_ref_mode=op.vap_ref_mode, slip_mode=op.slip_mode)
        fl = iki_fazli_akisi_coz(st, op.P2_pa, Cd, orif.Cc, orif.L_orifis_mm * 1e-3, op.slip_mode, kal)
        G = {k: fl.G_kal[k] for k in ('dual_spc', 'hem_c', 'fml', 'dyer_comp')}
        G['omega'] = omega_akisi(P1, x1, op.P2_pa, Cd)[0]
        return st, fl, G

    st3, fl3, G3 = _iki_faz(x1_bilinen)
    md3 = {k: g * A for k, g in G3.items()}
    dg3 = {k: boyut_hesapla_mm(m_h, g, orif) for k, g in G3.items()}
    man3 = manifold_coz(m_h, st3, feed.manifold, orif, d_mm, fl3, op.P2_pa) if (feed.manifold is not None and orif.annulus) else None
    w3, ek3, ch3 = omega_akisi(P1, x1_bilinen, op.P2_pa, Cd)[1:]
    n3 = [f"Giriş P1 = {P1/1e5:.2f} bar'da doymuş (T1 = {st3.Tsat_P_K-273.15:.1f} °C), x1 = %{x1_bilinen*100:.1f}, "
          f"giriş void = %{st3.alpha*100:.0f}. Hat kaybı bağlaşımı yok (P1 sabit).",
          f"Leung ω = {w3:.2f}, kritik basınç oranı {ek3:.3f} → {'boğulmuş' if ch3 else 'boğulmamış'}.",
          "Nurick doymuş girişte geçersiz (K ≈ 0), dahil edilmedi."]
    out.append(_paket('S3', f'Karışık giriyor (x1 = %{x1_bilinen*100:.0f} biliniyor), karışık çıkıyor',
                      f"Doymuş karışım, {P1/1e5:.2f} bar, x1 = %{x1_bilinen*100:.1f}", 'dual_spc', md3, dg3, man3, [], n3))

    tarama = []
    for x1 in x1_liste:
        st4, fl4, G4 = _iki_faz(float(x1))
        row = {'x1': float(x1), 'mdot': {k: G4[k] * A for k in ('dual_spc', 'hem_c', 'omega')},
               'd': {k: boyut_hesapla_mm(m_h, G4[k], orif) for k in ('dual_spc', 'hem_c', 'omega')}}
        if feed.manifold is not None and orif.annulus:
            ms = manifold_coz(m_h, st4, feed.manifold, orif, d_mm, fl4, op.P2_pa)
            row.update(alpha_man=ms.alpha, V_buhar_cm3=ms.V_buhar_cm3, f_h=ms.f_helmholtz_Hz, a_ses=ms.a_ses_m_s)
        tarama.append(row)
    md4 = {f"{k}@{r['x1']:.2f}": r['mdot'][k] for r in tarama for k in r['mdot']}
    dg4 = {f"{k}@{r['x1']:.2f}": r['d'][k] for r in tarama for k in r['d']}
    n4 = [f"x1 = %{min(x1_liste)*100:.0f}..%{max(x1_liste)*100:.0f} taraması, Dual-SPC / HEMc / Leung ω. Sonuç tek sayı değil zarf.",
          "Bu durumun ölçümle ayrılması için enjektör girişinde PT + TC önerilir: P1 > Pv(T1) ise sıvı (S1/S2)."]
    s4 = _paket('S4', 'Karışık giriyor (x1 bilinmiyor), karışık çıkıyor', f"Doymuş karışım, {P1/1e5:.2f} bar, x1 taraması",
                'dual_spc@0.00', md4, dg4, None, tarama, n4)
    out.append(s4)
    return out


def rapor_manifold(ms: ManifoldSonucu) -> None:
    print(f"\n[4b] MANİFOLD (V7.3)")
    print(_cizgi())
    print(f"  Hacim {ms.V_cm3:.1f} cm³ | N2O {ms.m_n2o_g:.1f} g | kalma süresi {ms.kalma_suresi_ms:.1f} ms")
    print(f"  Alanlar [mm²]: port {ms.A_port_mm2:.1f} | çevresel (tek yön) {ms.A_cevresel_mm2:.1f} | eksenel {ms.A_eksenel_mm2:.1f}"
          f" | annulus {ms.A_annulus_mm2:.1f} → çevresel/annulus {ms.oran_cevresel:.1f}, eksenel/annulus {ms.oran_eksenel:.1f}")
    print(f"  Hızlar [m/s]: port {ms.v_port_m_s:.2f} | çevresel {ms.v_cevresel_m_s:.2f} | eksenel {ms.v_eksenel_m_s:.3f}"
          f" | Re_çevresel {ms.Re_cevresel:.2e} | dağılım (q_çevresel/ΔP_enj) %{ms.dagilim_pct:.2f}")
    print(f"  Hal: {ms.P_bar:.2f} bar, {ms.T_C:.2f} °C, {ms.faz} | Pv {ms.Pv_bar:.2f} bar, Tsat(P) {ms.Tsat_C:.1f} °C"
          f" | alt-soğutma {ms.alt_sogutma_bar:.2f} bar / {ms.alt_sogutma_K:.1f} K | x %{ms.x*100:.1f}, void %{ms.alpha*100:.0f},"
          f" buhar {ms.V_buhar_cm3:.1f} cm³")
    print(f"  Kavitasyon: port σ = {ms.sigma_port:.2f} → {'KAVİTASYON' if ms.port_kavitasyon else 'OK'} | annulus girişi Nurick K = "
          f"{ms.annulus_K:.3f} / K_kr {ms.annulus_K_kr:.2f} → {'KAVİTASYONLU' if ms.annulus_kavitasyon else 'kavitasyonsuz'}"
          f" | çıkışta flaş: {'EVET' if ms.annulus_flas else 'hayır'} (κ {ms.kappa_dyer:.2f}, τ_r {ms.tau_r_us:.0f} µs,"
          f" τ_b {ms.tau_b_us:.0f} µs)")
    print(f"  Dinamik: ses hızı {ms.a_ses_m_s:.0f} m/s | esneklik {ms.C_cm3_bar:.4f} cm³/bar | Helmholtz ~{ms.f_helmholtz_Hz:.0f} Hz"
          f" | kabarcık yükselme {ms.u_kabarcik_m_s:.2f} m/s → {'BİRİKME RİSKİ' if ms.kabarcik_birikme else 'süpürülür'}")
    for w in ms.uyarilar:
        print(f"    [!] {w}")


def rapor_sicaklik(rows: List[SicaklikSatiri], op: OperatingConfig, orif: OrificeConfig) -> None:
    bad = 'g' if orif.annulus else 'd'
    print(f"\n[9] SICAKLIK TARAMASI (tank {op.P_tank_pa/1e5:.1f} bar sabit, P2 {op.P2_pa/1e5:.1f} bar, tasarım {op.tasarim_modeli})")
    print(_cizgi())
    print(f"  {'T °C':>5} | {'Pv bar':>6} | {'P_man':>6} | {'altS bar':>8} | {'ṁ tas.':>7} | {'sapma':>7} | {'O/F':>5} | "
          f"{'O/F bant':>11} | {bad + ' gerek.':>8} | {bad + ' bant':>11} | {'K':>5} | Durum")
    for r in rows:
        if not r.mdot:
            print(f"  {r.T_C:>5.1f} | {r.Pv_bar:>6.2f} | {r.P_man_bar:>6.2f} | {r.alt_sogutma_tank_bar:>8.2f} | " + "-" * 7 + f" | {r.durum}")
            continue
        dt = r.d_gerekli.get(op.tasarim_modeli, float('nan'))
        print(f"  {r.T_C:>5.1f} | {r.Pv_bar:>6.2f} | {r.P_man_bar:>6.2f} | {r.alt_sogutma_man_bar:>8.2f} | "
              f"{r.mdot.get(op.tasarim_modeli, float('nan')):>7.3f} | {r.sapma_pct:>+6.1f}% | {r.OF_tasarim:>5.2f} | "
              f"{r.OF_min:>5.2f}-{r.OF_max:<5.2f} | {dt:>8.3f} | {r.d_bant_min:>5.3f}-{r.d_bant_max:<5.3f} | {r.K_nurick:>5.2f} | {r.durum}")


def rapor_senaryo(sen: List[SenaryoSonucu], op: OperatingConfig, orif: OrificeConfig) -> None:
    print(f"\n[10] SENARYOLAR (sabit {orif.boyut_adi}, hedef {op.m_dot_hedef:.3f} kg/s)")
    print(_cizgi())
    for s in sen:
        print(f"  {s.kod} {s.ad} | {s.giris}")
        print(f"     ṁ {s.mdot_min:.3f}..{s.mdot_max:.3f} kg/s | O/F {s.OF_min:.2f}..{s.OF_max:.2f} | {s.kabul}")
        if s.kod != 'S4':
            for k, v in s.mdot.items():
                print(f"       {SENARYO_MODEL_ADLARI.get(k, k):<44}: ṁ {v:.3f} | O/F {s.OF[k]:.2f} | gereken {s.d_gerekli[k]:.3f} mm")
        else:
            for r in s.tarama:
                print(f"       x1 %{r['x1']*100:>4.1f}: " + " | ".join(f"{k} ṁ {v:.3f}" for k, v in r['mdot'].items())
                      + (f" | manifold void %{r['alpha_man']*100:.0f}, f_H ~{r['f_h']:.0f} Hz" if 'alpha_man' in r else ""))
        for n in s.notlar:
            print(f"       * {n}")


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
                 kal_ozet: Dict[str, Tuple[float, float, int]], vv_yazdir: bool = True,
                 man: Optional['ManifoldSonucu'] = None, tarama: Optional[List['SicaklikSatiri']] = None,
                 senaryolar: Optional[List['SenaryoSonucu']] = None) -> None:
    st, fl, b, fd, t = sol.st_tank, sol.fl, sol.boy, sol.fd, sol.th
    bad = orif.boyut_adi
    BAD = 'ARALIK' if orif.annulus else 'ÇAP'
    geo_str = (f"ANNULUS (iç çap {orif.D_ic_mm:.2f} mm)" if orif.annulus else f"ORİFİS (N = {orif.N_orifis})")
    si = sol.st_in
    tm = op.tasarim_modeli
    P2b = op.P2_pa / 1e5
    print("\n" + _cizgi('='))
    print(f"   N2O ENJEKTÖR AKIŞ / BESLEME / AYNA TERMAL RAPORU  (V7.3)  -  {geo_str}")
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
    if man is not None:
        rapor_manifold(man)
    if tarama:
        rapor_sicaklik(tarama, op, orif)
    if senaryolar:
        rapor_senaryo(senaryolar, op, orif)
    print(_cizgi('=') + "\n")


# %% [14] ÇALIŞTIRMA  (TÜM GİRDİLER BURADA)
# Kendi soğuk akış testleriniz. Orifis testi:
#   KalibrasyonTesti(P1_bar=55.0, T1_K=290.0, P2_bar=1.013, d_mm=2.20, N_orifis=8, mdot_olculen=0.95)
# Annulus testi (d_mm = aralık, N_orifis = 1, D_ic_mm = iç çap):
#   KalibrasyonTesti(P1_bar=55.0, T1_K=290.0, P2_bar=1.013, d_mm=0.88, N_orifis=1, mdot_olculen=0.95, D_ic_mm=14.0)
KALIBRASYON_TESTLERI: List[KalibrasyonTesti] = []

# Hangi senaryolar çalışsın: 'orifis', 'annulus' veya ikisi birden
SENARYOLAR = ['annulus']

if __name__ == "__main__":
    # ---- Ortak işletme girdileri -------------------------------------------------
    # Kesinleşenler: P2 = 27 bar; N2O helyumla basınçlandırılır, tank 62 bar / 20 °C.
    op = OperatingConfig(
        m_dot_rpa=1.09608, eta_atomizasyon=0.90,
        P_tank_pa=62.0e5, T_tank_K=293.15, P2_pa=27.0e5,
        manuel_x1=None,
        exponent_mode='real_ns', vap_ref_mode='T1_sat', slip_mode='zivi',
        tasarim_modeli='dual_spc',
    )
    manifold = ManifoldConfig(
        D_ic_mm=14.0, D_dis_mm=44.0, L_mm=8.0,       # <- GAZAP: r 7-22 mm, 8 mm derin (~11 cm³)
        N_giris=2, d_giris_mm=10.0,                  # <- giriş portları (toplam alan ~ hat alanı)
        K_giris=1.0, K_donus=0.5, sigma_i=2.0,
    )
    feed_orifis = FeedSystemConfig(
        D_hat_mm=15.7, L_hat_m=1.0, eps_boru_mm=0.015,
        K_tank_cikis=0.5, Cv_ana_vana=5.5, FL_vana=0.60, K_fitting_toplam=0.8,
        K_manifold_giris=1.5, D_manifold_ic_mm=60.0, H_manifold_ic_mm=10.0,
    )
    feed_annulus = replace(feed_orifis, manifold=manifold)
    th = ThermalConfig(
        T_gaz_eff_low_K=1400.0, T_gaz_eff_nom_K=2050.0, T_gaz_eff_high_K=2453.0,
        h_gaz_low=500.0, h_gaz_nominal=950.0, h_gaz_high=1600.0,
        D_ayna_mm=83.4, D_merkez_mm=20.0,            # <- gerçek ölçülerle değiştir
        malzeme='AISI_316L', t_yanma_s=5.0,
        t_kapanma_sonrasi_s=30.0, h_n2o_durgun=300.0,
        V_hapsolan_cm3=25.0,                         # <- vana-ayna arası gerçek hacim
        f_mix_nominal=0.35,
        T_wall_caution_K=473.15, T_wall_hazard_K=573.15,
    )
    OF_RPA, OF_BANT = 4.0, 10.0                      # O/F: hedef debide RPA değeri, kabul bandı ±%
    SICAKLIKLAR = [round(x, 1) for x in np.arange(0.0, 40.01, 2.5)]

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
        D_ic_mm=14.0,                  # <- halkanın iç çapı [mm] (GAZAP: pintle dış çapı)
        L_orifis_mm=8.0,               # <- halka kanal boyu [mm]
        Cd_tekfaz=0.75,                # <- KAVİTASYONSUZ tek faz Cd; su testiyle ölçün
        inlet_edge_type='sharp', Cc_override=None,
        drill_tolerance_mm=0.02,       # <- aralık toleransı [±mm]
        eksantriklik_mm=0.02,          # <- iç/dış merkez kaçıklığı [mm]
    )
    araliklar = [round(x, 2) for x in np.arange(0.60, 1.101, 0.05)]       # taranacak aralıklar [mm]

    # ---- Çalıştırma (bu satırların altına dokunmayın) ----------------------------
    vv = vv_makale_karsilastirmasi()
    senaryo_tanim = {'orifis': (orif_orifis, matkaplar, feed_orifis), 'annulus': (orif_annulus, araliklar, feed_annulus)}
    for i, ad in enumerate(SENARYOLAR):
        orif, boyutlar, feed = senaryo_tanim[ad]
        kal_ozet = kalibrasyon_hesapla(KALIBRASYON_TESTLERI, op, orif) if KALIBRASYON_TESTLERI else {}
        kal = {k: v[0] for k, v in kal_ozet.items()}
        sol = baglasik_coz(op, orif, feed, th, kal)
        tablo = matkap_tablosu_coz(sol, op, orif, feed, kal, boyutlar)
        uq = belirsizlik_coz(sol, op, orif, kal)
        d_sabit = sol.boy.d_tasarim_mm
        man = (manifold_coz(op.m_dot_hedef, sol.st_in, feed.manifold, orif, d_sabit, sol.fl, op.P2_pa)
               if (feed.manifold is not None and orif.annulus) else None)
        tarama = sicaklik_taramasi(op, orif, feed, kal, d_sabit, SICAKLIKLAR, OF_RPA)
        sen = senaryo_coz(op, orif, feed, kal, sol, d_sabit, OF_RPA, OF_BANT) if orif.annulus else None
        rapor_yazdir(op, orif, feed, th, vv, sol, tablo, uq, kal_ozet, vv_yazdir=(i == 0),
                     man=man, tarama=tarama, senaryolar=sen)
