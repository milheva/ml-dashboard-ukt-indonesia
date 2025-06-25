import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- Konfigurasi Halaman dan Styling (CSS Kustom) ---
st.set_page_config(
    page_title="Dashboard UKT Jalur Mandiri", page_icon="🏦", layout="wide"
)

# CSS Kustom untuk mempercantik tampilan
st.markdown(
    """
<style>
    /* Mengubah font utama */
    html, body, [class*="css"]  {
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Styling untuk kontainer utama */
    .st-emotion-cache-1r4qj8v {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: calc(1em - 1px);
    }
    
    /* Styling tombol utama (Prediksi Biaya) */
    .stButton>button {
        border-radius: 10px;
        color: #ffffff;
        background-color: #0068c9;
        border: 2px solid #0068c9;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #0068c9;
        border: 2px solid #0068c9;
    }
    
    /* ===============================================================
    SEMUA STYLING KHUSUS UNTUK SIDEBAR SENGAJA DIHAPUS 
    AGAR KEMBALI KE TAMPILAN DEFAULT (LATAR PUTIH, TEKS HITAM).
    INI AKAN MENYELESAIKAN MASALAH VISIBILITAS TOMBOL.
    ===============================================================
    */

</style>
""",
    unsafe_allow_html=True,
)


# --- Fungsi-fungsi Backend (ditaruh di cache agar cepat) ---


def clean_currency(value):
    if isinstance(value, str):
        value = re.sub(r"Rp|\.|\s", "", value)
        if value == "-" or value == "":
            return np.nan
        return pd.to_numeric(value, errors="coerce")
    return value


@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=";")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.drop(columns=["NO", "Referensi"], errors="ignore")

    ukt_cols = [col for col in df.columns if "UKT GOL" in col]
    for col in ukt_cols:
        df[col] = df[col].apply(clean_currency)

    df["UKT_MAKSIMAL"] = df[ukt_cols].max(axis=1)
    df.dropna(subset=["UKT_MAKSIMAL", "PTN", "PROGRAM STUDI", "PULAU"], inplace=True)

    features = ["PTN", "PROGRAM STUDI", "PULAU", "DAYA TAMPUNG"]
    target = "UKT_MAKSIMAL"

    X = df[features].copy()
    y = df[target]

    if X["DAYA TAMPUNG"].isnull().any():
        X["DAYA TAMPUNG"].fillna(X["DAYA TAMPUNG"].median(), inplace=True)

    X["DAYA TAMPUNG"] = X["DAYA TAMPUNG"].astype(int)

    encoders = {}
    for col in ["PTN", "PROGRAM STUDI", "PULAU"]:
        le = LabelEncoder()
        all_labels = pd.concat([X[col], X[col].astype("str")]).unique()
        le.fit(all_labels)
        X[col] = le.transform(X[col].astype("str"))
        encoders[col] = le

    return df, X, y, encoders, features


# --- Memuat Data ---
try:
    df, X, y, encoders, features = load_and_preprocess_data("DATA VINIX Program.csv")
except Exception as e:
    st.error(f"Gagal memuat atau memproses data: {e}")
    st.info(
        "Pastikan file 'DATA VINIX Program.csv' ada di folder yang sama dan formatnya benar."
    )
    st.stop()


# --- BAGIAN UTAMA: Tampilan Dashboard ---

st.title("🏦 Dashboard Analisis & Prediksi Biaya UKT Jalur Mandiri")

st.write(
    """
**Oleh: I Wayan Indra Sakti Sanjaya** | Dashboard ini mengintegrasikan wawasan dari **Data Mining**, kekuatan prediksi **Machine Learning**, dan simulasi interaktif berbasis **Information Retrieval**.
"""
)

# --- BAGIAN 1: DATA MINING ---
st.header("📊 1. Analisis Data (Data Mining)")
with st.container(border=True):
    st.subheader("Eksplorasi Data Referensi UKT Jalur Mandiri")
    st.write(
        "Bagian ini menampilkan wawasan dari data referensi untuk memahami karakteristik dan distribusi biaya pendidikan pada Jalur Mandiri di Perguruan Tinggi Negeri Indonesia."
    )

    with st.expander("Tampilkan Data yang Telah Diproses"):
        display_df = df.copy()
        ukt_display_cols = [col for col in display_df.columns if "UKT" in col]
        for col in ukt_display_cols:
            display_df[col] = display_df[col].apply(
                lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "-"
            )
        st.dataframe(display_df, use_container_width=False)

    st.subheader("Program Studi dengan UKT Maksimal Tertinggi (Jalur Mandiri)")
    top_10_prodi = df.nlargest(10, "UKT_MAKSIMAL").sort_values(
        "UKT_MAKSIMAL", ascending=False
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        x="UKT_MAKSIMAL",
        y="PROGRAM STUDI",
        data=top_10_prodi,
        ax=ax,
        palette="plasma",
        hue="PTN",
        dodge=False,
    )
    ax.set_title("Top 10 Program Studi dengan UKT Maksimal Tertinggi", fontsize=16)
    ax.set_xlabel("UKT Maksimal (dalam Juta Rupiah)", fontsize=12)
    ax.set_ylabel("Program Studi", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.1f} Jt"))
    plt.legend(title="PTN", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig, use_container_width=True)


# --- BAGIAN 2: MACHINE LEARNING ---
st.header("🤖 2. Prediksi UKT Maksimal (Machine Learning)")
with st.container(border=True):
    st.subheader("Melatih Model untuk Memprediksi Biaya")

    with st.expander("Klik di sini untuk penjelasan cara kerja Algoritma"):
        st.markdown(
            """
        #### Bagaimana Cara Kerja Algoritma Random Forest?
        Untuk memprediksi UKT, kita menggunakan algoritma **Random Forest Regressor**. Bayangkan algoritma ini seperti sebuah **rapat dewan para ahli** untuk menebak harga.
        1.  **Banyak Ahli (Pohon Keputusan):** Alih-alih hanya memiliki satu ahli, Random Forest menciptakan ratusan "ahli" yang disebut *Decision Tree* (Pohon Keputusan).
        2.  **Latihan yang Berbeda:** Setiap ahli (pohon) hanya dilatih pada sebagian data yang dipilih secara acak. Selain itu, setiap kali membuat keputusan, mereka hanya boleh mempertimbangkan beberapa fitur acak.
        3.  **Mencegah Bias:** Proses acak ini bertujuan agar setiap ahli memiliki "spesialisasi" yang berbeda dan tidak semuanya membuat kesalahan yang sama. Ini membuat model menjadi sangat kuat dan tidak mudah "tertipu" oleh data yang aneh (overfitting).
        4.  **Musyawarah untuk Mufakat:** Saat kita meminta prediksi baru, setiap ahli akan memberikan tebakannya sendiri.
        5.  **Hasil Akhir (Prediksi):** Jawaban akhir dari Random Forest bukanlah suara mayoritas, melainkan **rata-rata dari semua tebakan para ahli tersebut**. Dengan merata-ratakan banyak tebakan yang beragam, hasilnya menjadi jauh lebih akurat dan stabil.
        """
        )

    # Proses training dan evaluasi model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    oob_score = model.oob_score_

    st.subheader("Hasil Performa Model")
    st.write("Model dievaluasi untuk melihat seberapa akurat prediksinya.")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "R-squared (R²)",
        f"{r2:.3f}",
        help="Seberapa besar variasi biaya UKT yang bisa dijelaskan oleh fitur. Semakin dekat ke 1, semakin baik.",
    )
    col2.metric(
        "Mean Absolute Error (MAE)",
        f"Rp {mae:,.0f}",
        help="Rata-rata kesalahan absolut prediksi.",
    )
    col3.metric(
        "Out-of-Bag (OOB) Score",
        f"{oob_score:.3f}",
        help="Estimasi R² model pada data yang tidak terlihat saat training.",
    )

    st.subheader("Faktor Paling Berpengaruh (Feature Importance)")
    st.write(
        "Grafik ini menunjukkan faktor apa yang dianggap paling penting oleh model saat membuat prediksi."
    )
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x="importance", y="feature", data=importance, ax=ax, palette="viridis")
    ax.set_title("Tingkat Kepentingan Fitur")
    st.pyplot(fig)


# --- BAGIAN 3: INFORMATION RETRIEVAL (Sidebar) ---
st.sidebar.header("🔍 Simulasi UKT Jalur Mandiri")

# Dropdown Berantai
unique_pulau = sorted(df["PULAU"].dropna().unique())
selected_pulau = st.sidebar.selectbox("Langkah 1: Pilih Pulau", options=unique_pulau)

ptn_in_pulau = sorted(df[df["PULAU"] == selected_pulau]["PTN"].unique())
selected_ptn = st.sidebar.selectbox("Langkah 2: Pilih PTN", options=ptn_in_pulau)

prodi_in_ptn = sorted(df[df["PTN"] == selected_ptn]["PROGRAM STUDI"].unique())
selected_prodi = st.sidebar.selectbox(
    "Langkah 3: Pilih Program Studi", options=prodi_in_ptn
)

default_daya_tampung = df[
    (df["PTN"] == selected_ptn) & (df["PROGRAM STUDI"] == selected_prodi)
]["DAYA TAMPUNG"].values
default_value = (
    int(default_daya_tampung[0])
    if len(default_daya_tampung) > 0
    else int(df["DAYA TAMPUNG"].median())
)
daya_tampung_input = st.sidebar.number_input(
    "Langkah 4: Masukkan Daya Tampung", value=default_value
)

with st.sidebar.expander("Mengapa 'Daya Tampung' Penting?"):
    st.markdown(
        """
    `Daya Tampung` adalah jumlah kursi yang tersedia. Fitur ini penting bagi model karena bisa mencerminkan prinsip **penawaran dan permintaan (supply and demand)**.
    """
    )


if st.sidebar.button("Prediksi Biaya UKT Maksimal"):
    user_inputs = {
        "PULAU": selected_pulau,
        "PTN": selected_ptn,
        "PROGRAM STUDI": selected_prodi,
        "DAYA TAMPUNG": daya_tampung_input,
    }
    input_df = pd.DataFrame([user_inputs])
    for col in ["PTN", "PROGRAM STUDI", "PULAU"]:
        input_df[col] = encoders[col].transform(input_df[col])

    input_df = input_df[X.columns]

    with st.spinner("Mencari prediksi biaya..."):
        prediction = model.predict(input_df)

    st.sidebar.success(f"**Prediksi UKT Maksimal:**")
    st.sidebar.markdown(
        f"<h3 style='text-align: center; color: green;'>Rp {prediction[0]:,.0f}</h3>",
        unsafe_allow_html=True,
    )
