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
    
    /* Mengatur agar sidebar kembali ke style default (putih) */

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

# --- BAGIAN 1: DATA MINING (DIREVISI TOTAL) ---
st.header("📊 1. Analisis Data (Data Mining)")
with st.container(border=True):
    st.subheader("Eksplorasi Data Referensi UKT Jalur Mandiri")
    st.write(
        "Bagian ini menampilkan berbagai wawasan dari data untuk memahami karakteristik dan distribusi biaya pendidikan pada Jalur Mandiri di Perguruan Tinggi Negeri Indonesia."
    )

    with st.expander("Tampilkan Data Mentah yang Telah Diproses"):
        st.dataframe(df, use_container_width=False)

    st.markdown("---")
    st.subheader("Visualisasi Data Eksploratif")

    # Membuat layout kolom 2x2 untuk 4 grafik
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Grafik 1: Program Studi Termahal (di kolom 1)
    with col1:
        st.write("**10 Program Studi dengan UKT Termahal**")
        top_10_prodi = df.nlargest(10, "UKT_MAKSIMAL")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x="UKT_MAKSIMAL",
            y="PROGRAM STUDI",
            data=top_10_prodi,
            ax=ax1,
            palette="plasma",
            hue="PTN",
            dodge=False,
        )
        ax1.get_legend().remove()  # Hapus legend duplikat agar rapi
        ax1.set_xlabel("UKT Maksimal (Juta Rupiah)")
        ax1.set_ylabel("")
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}"))
        st.pyplot(fig1, use_container_width=True)
        st.caption(
            "Grafik menyorot program studi yang menjadi outlier dari segi biaya, didominasi oleh bidang Kedokteran."
        )

    # Grafik 2: Distribusi Keseluruhan UKT (di kolom 2)
    with col2:
        st.write("**Distribusi Keseluruhan Biaya UKT**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(df["UKT_MAKSIMAL"], kde=True, ax=ax2, bins=30, color="darkcyan")
        ax2.set_xlabel("UKT Maksimal (Juta Rupiah)")
        ax2.set_ylabel("Jumlah Program Studi")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}"))
        st.pyplot(fig2, use_container_width=True)
        st.caption(
            "Sebaran biaya UKT cenderung 'right-skewed', artinya mayoritas prodi memiliki UKT di bawah 25 juta, namun ada beberapa prodi dengan biaya sangat tinggi."
        )

    # Grafik 3: Perbandingan UKT per Pulau (di kolom 3)
    with col3:
        st.write("**Perbandingan UKT Maksimal per Pulau**")
        # Mengurutkan pulau berdasarkan median UKT untuk insight yang lebih baik
        order = (
            df.groupby("PULAU")["UKT_MAKSIMAL"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            x="PULAU", y="UKT_MAKSIMAL", data=df, ax=ax3, palette="crest", order=order
        )
        ax3.set_xlabel("")
        ax3.set_ylabel("UKT Maksimal (Juta Rupiah)")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}"))
        plt.xticks(rotation=45)
        st.pyplot(fig3, use_container_width=True)
        st.caption(
            "Box plot menunjukkan bahwa median dan sebaran biaya UKT di Pulau Jawa secara signifikan lebih tinggi dibandingkan pulau lainnya."
        )

    # Grafik 4: Hubungan Daya Tampung vs UKT (di kolom 4)
    with col4:
        st.write("**Hubungan Daya Tampung & Biaya UKT**")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x="DAYA TAMPUNG",
            y="UKT_MAKSIMAL",
            data=df,
            ax=ax4,
            alpha=0.3,
            color="indigo",
        )
        ax4.set_xlabel("Daya Tampung")
        ax4.set_ylabel("UKT Maksimal (Juta Rupiah)")
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.0f}"))
        st.pyplot(fig4, use_container_width=True)
        st.caption(
            "Diagram pencar ini menunjukkan tidak ada korelasi linear yang jelas antara daya tampung dan biaya UKT. Program mahal bisa memiliki daya tampung besar maupun kecil."
        )

# --- BAGIAN 2: MACHINE LEARNING ---
st.header("🤖 2. Prediksi UKT Maksimal (Machine Learning)")
with st.container(border=True):
    # (Kode di bagian ini tidak berubah, tetap sama seperti sebelumnya)
    st.subheader("Melatih Model untuk Memprediksi Biaya")
with st.expander("Klik di sini untuk penjelasan cara kerja Algoritma"):
        st.markdown(
            """
        #### Mengenal Cara Kerja Random Forest Regressor

        Algoritma yang menjadi inti dari fitur prediksi ini adalah **Random Forest Regressor**. Prinsip kerjanya adalah "kekuatan dalam jumlah" atau yang dikenal sebagai *ensemble learning*.

        Daripada membangun satu model tunggal yang kompleks, Random Forest membangun ratusan model yang lebih sederhana (*Decision Tree*) secara independen. Setiap model ini "belajar" dari porsi data yang dipilih secara acak. Selain itu, dalam setiap tahap pembelajarannya, setiap pohon hanya mempertimbangkan sebagian kecil dari total fitur yang ada (misalnya hanya `Pulau` dan `Daya Tampung`).

        Proses pengacakan ganda ini—baik pada data maupun pada fitur—memastikan bahwa setiap pohon memiliki "spesialisasinya" sendiri dan tidak saling meniru kesalahan satu sama lain.

        Saat prediksi dilakukan, setiap pohon akan memberikan prediksinya sendiri. Hasil akhir dari Random Forest bukanlah memilih prediksi dari pohon terbaik, melainkan **agregat (dalam kasus ini, rata-rata) dari semua prediksi pohon tersebut**. Metode ini terbukti sangat efektif untuk menghasilkan prediksi yang akurat dan tahan terhadap data yang tidak biasa (*outlier*).
        """
        )
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
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x="importance", y="feature", data=importance, ax=ax, palette="viridis")
    ax.set_title("Tingkat Kepentingan Fitur")
    st.pyplot(fig)


# --- BAGIAN 3: INFORMATION RETRIEVAL (Sidebar) ---
st.sidebar.header("🔍 Simulasi UKT Jalur Mandiri")
# (Kode di bagian ini tidak berubah, tetap sama seperti sebelumnya)
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
        "`Daya Tampung` adalah jumlah kursi yang tersedia. Fitur ini penting bagi model karena bisa mencerminkan prinsip **penawaran dan permintaan (supply and demand)**."
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
