import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="SA-5 Feed-Forward + Noise Filtering", page_icon="🎛️")

st.title("🎛️ SA-5 – Feed-Forward + Noise Filtering Lab")
st.write(
    """
Bu laboratuvarda PID'li oda ısıtma sistemine iki zorluk ekliyoruz:

1. Dış bozulma: Belirli bir anda **pencere açılıyor**, oda hızlı soğuyor.  
2. Sensör gürültüsü: Sıcaklık ölçümü küçük rastgele hatalar içeriyor.

Sonra iki savunma mekanizmasını test ediyoruz:

- **Feed-forward (FF):** Bozulma anında ısıtıcı gücünü önceden artır.  
- **Noise filtering:** Gürültülü ölçümü filtreleyerek PID'in daha sakin davranmasını sağla.
"""
)

st.markdown("---")


# -----------------------------
# Sistem ve PID parametreleri
# -----------------------------
st.subheader("1️⃣ Sistem ve PID Ayarları")

col_sys1, col_sys2, col_sys3 = st.columns(3)
with col_sys1:
    T_ambient = st.slider(
        "Ortam sıcaklığı (°C)",
        0.0,
        30.0,
        20.0,
        1.0,
    )
with col_sys2:
    T_set = st.slider(
        "Setpoint (hedef sıcaklık, °C)",
        15.0,
        30.0,
        24.0,
        0.5,
    )
with col_sys3:
    tau = st.slider(
        "Zaman sabiti τ (s)",
        10.0,
        200.0,
        60.0,
        10.0,
    )

k_heat = st.slider(
    "Isıtıcı kazancı k_heat",
    0.1,
    2.0,
    0.5,
    0.1,
)

col_pid1, col_pid2, col_pid3 = st.columns(3)
with col_pid1:
    Kp = st.slider("Kp", 0.0, 10.0, 3.0, 0.1)
with col_pid2:
    Ki = st.slider("Ki", 0.0, 1.0, 0.2, 0.01)
with col_pid3:
    Kd = st.slider("Kd", 0.0, 2.0, 0.0, 0.1)

st.write(
    f"Sistem: ortam **{T_ambient:.1f}°C**, hedef **{T_set:.1f}°C**, "
    f"τ = **{tau:.0f} s**, k_heat = **{k_heat:.2f}**; "
    f"PID: **Kp = {Kp:.2f}**, **Ki = {Ki:.2f}**, **Kd = {Kd:.2f}**"
)


# -----------------------------
# Bozulma (disturbance) ve gürültü
# -----------------------------
st.subheader("2️⃣ Bozulma (Pencere Açma) ve Sensör Gürültüsü")

col_d1, col_d2 = st.columns(2)
with col_d1:
    t_disturb = st.slider(
        "Pencere açılma zamanı (s)",
        0.0,
        300.0,
        100.0,
        10.0,
    )
with col_d2:
    disturb_strength = st.slider(
        "Bozulma şiddeti D (°C/s civarı soğutma etkisi)",
        0.0,
        1.0,
        0.3,
        0.05,
        help="D ne kadar büyükse pencere açıldığında oda o kadar hızlı soğur.",
    )

noise_level = st.slider(
    "Sensör gürültü seviyesi",
    0.0,
    1.0,
    0.2,
    0.05,
    help="0: gürültü yok, 1: oldukça gürültülü sensör.",
)

st.markdown(
    "Bozulma, pencere açıldığında sıcaklık değişim hızına eklenen **negatif** bir terim gibi düşünülebilir."
)


# -----------------------------
# Feed-forward ve filtre ayarları
# -----------------------------
st.subheader("3️⃣ Feed-Forward ve Ölçüm Filtresi")

col_ff1, col_ff2 = st.columns(2)
with col_ff1:
    use_ff = st.checkbox("Feed-forward kullan", value=False)
with col_ff2:
    ff_gain = st.slider(
        "Feed-forward kazancı (k_ff)",
        0.0,
        2.0,
        1.0,
        0.1,
        help="1.0 civarı, bozulmayı yaklaşık dengeleyecek bir FF sağlar.",
    )

col_filt1, col_filt2 = st.columns(2)
with col_filt1:
    use_filter = st.checkbox("Sensör verisine filtre uygula", value=True)
with col_filt2:
    alpha = st.slider(
        "Üstel filtre katsayısı α",
        0.1,
        1.0,
        0.3,
        0.05,
        help="T_filt = α * T_meas + (1-α) * T_filt_prev; α küçükse daha pürüzsüz ama daha gecikmeli.",
    )

st.caption(
    "Not: FF bozulma anında ekstra ısı verir. Filtre ise gürültülü ölçümü yumuşatarak PID'in daha az zıplamasını sağlar."
)


# -----------------------------
# Simülasyon ayarları
# -----------------------------
st.subheader("4️⃣ Simülasyon Ayarları")

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    T_initial = st.slider(
        "Başlangıç sıcaklığı T₀ (°C)",
        0.0,
        30.0,
        20.0,
        0.5,
    )
with col_sim2:
    t_max = st.slider(
        "Toplam süre (s)",
        60.0,
        600.0,
        300.0,
        30.0,
    )

dt = st.slider(
    "Zaman adımı Δt (s)",
    0.1,
    5.0,
    1.0,
    0.1,
)

n_steps = int(t_max / dt) + 1
st.write(f"Simülasyon: {t_max:.0f} s, Δt = {dt:.1f} s, adım ≈ {n_steps}")


# -----------------------------
# Simülasyon fonksiyonu
# -----------------------------
def simulate_ff_filter(
    T_ambient,
    T_set,
    T_initial,
    tau,
    k_heat,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
    t_disturb,
    disturb_strength,
    noise_level,
    use_ff,
    ff_gain,
    use_filter,
    alpha,
    seed=0,
):
    rng = np.random.default_rng(seed)

    t = np.zeros(n_steps)
    T_true = np.zeros(n_steps)
    T_meas = np.zeros(n_steps)
    T_filt = np.zeros(n_steps)
    u = np.zeros(n_steps)
    e = np.zeros(n_steps)
    d_hist = np.zeros(n_steps)

    T_true[0] = T_initial
    T_meas[0] = T_initial
    T_filt[0] = T_initial

    integral = 0.0
    prev_error = T_set - T_filt[0]

    for k in range(n_steps - 1):
        time = t[k]

        # Bozulma (pencere açılması): negatif soğutma etkisi
        if time >= t_disturb:
            d = -disturb_strength
        else:
            d = 0.0
        d_hist[k] = d

        # Sensör gürültüsü
        noise = noise_level * rng.standard_normal()
        T_meas[k] = T_true[k] + noise

        # Ölçüm filtresi
        if use_filter:
            if k == 0:
                T_filt[k] = T_meas[k]
            else:
                T_filt[k] = alpha * T_meas[k] + (1 - alpha) * T_filt[k - 1]
        else:
            T_filt[k] = T_meas[k]

        # PID
        error = T_set - T_filt[k]
        e[k] = error
        integral += error * dt
        derivative = (error - prev_error) / dt

        u_pid = Kp * error + Ki * integral + Kd * derivative

        # Feed-forward: bozulma varsa, ek ısı
        if use_ff and time >= t_disturb:
            # Disturbance büyüklüğüne göre yaklaşık dengeleme
            u_ff = ff_gain * (disturb_strength / k_heat) * 100.0
        else:
            u_ff = 0.0

        u_raw = u_pid + u_ff
        u[k] = np.clip(u_raw, 0.0, 100.0)

        # Oda modeli: dT/dt = -(T - T_amb)/tau + k_heat*(u/100) + d
        dTdt = -(T_true[k] - T_ambient) / tau + k_heat * (u[k] / 100.0) + d
        T_true[k + 1] = T_true[k] + dTdt * dt
        t[k + 1] = t[k] + dt

        prev_error = error

    # Son adım ölçümlerini doldur
    T_meas[-1] = T_true[-1] + noise_level * rng.standard_normal()
    if use_filter:
        T_filt[-1] = alpha * T_meas[-1] + (1 - alpha) * T_filt[-2]
    else:
        T_filt[-1] = T_meas[-1]
    e[-1] = T_set - T_filt[-1]
    u[-1] = u[-2]
    d_hist[-1] = d_hist[-2]

    return t, T_true, T_meas, T_filt, u, e, d_hist


# Simülasyonu çalıştır
t, T_true, T_meas, T_filt, u, e, d_hist = simulate_ff_filter(
    T_ambient,
    T_set,
    T_initial,
    tau,
    k_heat,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
    t_disturb,
    disturb_strength,
    noise_level,
    use_ff,
    ff_gain,
    use_filter,
    alpha,
)
# -----------------------------
# Grafikleri çiz
# -----------------------------
st.markdown("---")
st.subheader("5️⃣ Sıcaklık ve Setpoint – Gerçek / Ölçülen / Filtreli")

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(t, T_true, label="Gerçek sıcaklık T_true")
ax1.plot(t, T_meas, alpha=0.4, label="Ölçülen (gürültülü) T_meas")
ax1.plot(t, T_filt, label="Filtreli sıcaklık T_filt")
ax1.axhline(T_set, linestyle="--", color="gray", label="Setpoint")
ax1.axvline(t_disturb, linestyle=":", color="red", label="Pencere açılıyor")

ax1.set_xlabel("t (s)")
ax1.set_ylabel("Sıcaklık (°C)")
ax1.set_title("Bozulma ve Gürültü Altında Sıcaklık")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.legend()

st.pyplot(fig1)

st.subheader("Kontrol Sinyali u(t) ve Bozulma")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, u, label="u(t) – Isıtıcı gücü (%)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("u(t) (%)")
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

ax3 = ax2.twinx()
ax3.plot(t, d_hist, linestyle=":", color="red", label="Bozulma d(t)")
ax3.set_ylabel("d(t) (soğutma etkisi)")

# Legendleri birleştir
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper right")

st.pyplot(fig2)


# -----------------------------
# İlk adımlar tablosu
# -----------------------------
st.subheader("6️⃣ İlk Adımların Tablosu")

max_rows = min(20, n_steps)
df = pd.DataFrame(
    {
        "t (s)": t[:max_rows],
        "T_true": T_true[:max_rows],
        "T_meas": T_meas[:max_rows],
        "T_filt": T_filt[:max_rows],
        "u(t)": u[:max_rows],
        "d(t)": d_hist[:max_rows],
    }
)

st.dataframe(
    df.style.format(
        {
            "t (s)": "{:.1f}",
            "T_true": "{:.2f}",
            "T_meas": "{:.2f}",
            "T_filt": "{:.2f}",
            "u(t)": "{:.2f}",
            "d(t)": "{:.3f}",
        }
    )
)


# -----------------------------
# Öğretmen kutusu
# -----------------------------
st.markdown("---")
st.info(
    "Feed-forward, bozulmanın zamanını ve yönünü biliyorsak, ne kadar ekstra ısı "
    "vermemiz gerektiğini önceden tahmin etmeye çalışır. Noise filtering ise "
    "gürültülü ölçümü yumuşatarak PID çıktısının gereksiz zıplamalarını azaltır."
)

with st.expander("👩‍🏫 Öğretmen Kutusu – Önerilen Sorular (SA-5)"):
    st.write(
        """
**Feed-forward bölümü:**

1. Feed-forward **kapalı** iken (use_ff = False), bozulma anında sıcaklık grafiğine bak:
   - Pencere açıldığı anda (t = t_disturb) sıcaklık nasıl değişiyor?  
   - Hedefe geri dönme süresi ne kadar?

2. Aynı parametrelerle feed-forward **açık** (use_ff = True) iken:
   - Pencere açıldığı anda sıcaklıkta **düşüş** miktarı azaldı mı?  
   - Hedefe dönüş süresi kısaldı mı?

3. ff_gain parametresini 0.5, 1.0 ve 1.5 için karşılaştır:
   - Hangi değerde en iyi telafiyi gözlüyorsun?  
   - Aşırı büyük ff_gain durumda ne tür istenmeyen etkiler ortaya çıkabilir?

---

**Noise filtering bölümü:**

4. noise_level'i 0.8 gibi büyük bir değere ayarla.
   - Filtre **kapalıyken** (use_filter = False) T_meas ve u(t) grafikleri nasıl görünüyor?  
   - PID çıktısında gereksiz salınımlar var mı?

5. Aynı durumda filtreyi **aç** ve α değerini 0.3 civarına getir.
   - T_filt grafiği T_meas'e göre ne kadar daha pürüzsüz?  
   - u(t) daha sakin mi?

6. α'yı 0.9 ve 0.1 için karşılaştır:
   - α büyük → filtre hızlı ama daha az yumuşatıyor.  
   - α küçük → filtre çok yumuşak ama daha gecikmeli.  
   Gecikme ile gürültü azaltma arasında nasıl bir denge kurmak gerekir?
"""
    )

st.caption(
    "SA-5: Bu modül, feed-forward ve ölçüm filtresi gibi ek kontrol bloklarını "
    "PID çerçevesine sezgisel olarak eklemek için tasarlanmıştır."
)
