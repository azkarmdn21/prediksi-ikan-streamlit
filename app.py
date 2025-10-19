# PERHATIAN: Baris ini harus menjadi baris PERTAMA di sel kode Anda.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
import openpyxl 

warnings.filterwarnings('ignore')

# --- 1. FUNGSI PEMUATAN DAN PREDIKSI ---

@st.cache_data
def load_data():
    """Memuat data historis dan model yang telah dilatih."""
    try:
        # Muat model
        models_dict = joblib.load('all_models_hybrid.pkl')
    except FileNotFoundError:
        st.error("File model 'all_models_hybrid.pkl' tidak ditemukan. Pastikan file ada di folder yang sama.")
        st.stop()
        
    try:
        # Muat data historis (diperlukan untuk info tahun terakhir)
        df_hist = pd.read_excel('produksi_pembenihan_jawaBarat_2019_2023.xlsx') # Perbaikan file Excel
        df_hist['Tahun'] = pd.to_datetime(df_hist['Tahun'], format='%Y')
        
        # Lakukan agregasi yang sama seperti saat pelatihan
        data_grouped = df_hist.groupby(['Tahun', 'Kelompok Ikan']).agg({
            'Volume (Ribu Ekor)': 'sum',
            'Nilai (Rp. Juta)': 'sum',
            'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': 'mean'
        }).reset_index().set_index('Tahun')
        
    except FileNotFoundError:
        st.error("File data 'produksi_pembenihan_jawaBarat_2019_2023.xlsx' tidak ditemukan.")
        st.stop()
        
    return models_dict, data_grouped

def run_recursive_prediction(model_components, future_inputs_df):
    """
    Menjalankan prediksi hybrid secara rekursif.
    """
    
    # Ambil komponen model
    arima_result = model_components['arima_result']
    svr_model = model_components['svr_model']
    scaler = model_components['scaler']
    
    # Ambil lag volume terakhir dari data historis
    current_lag_volume = model_components['last_known_volume']
    
    n_periods = len(future_inputs_df)
    
    # 1. Dapatkan semua prediksi ARIMA sekaligus
    arima_preds = arima_result.forecast(steps=n_periods)
    
    # 2. Lakukan prediksi SVR satu per satu (rekursif)
    svr_preds = []
    hybrid_preds = []
    
    for i in range(n_periods):
        # Ambil input exogenous dari pengguna
        future_nilai = future_inputs_df.iloc[i]['Nilai (Rp. Juta)']
        future_harga = future_inputs_df.iloc[i]['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']
        
        # Buat fitur SVR: [Nilai, Harga, Volume_Lag1]
        X_svr_future = pd.DataFrame({
            'Nilai (Rp. Juta)': [future_nilai],
            'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': [future_harga],
            'Volume_Lag1': [current_lag_volume]
        })
        
        # Scaling
        X_svr_scaled = scaler.transform(X_svr_future)
        
        # Prediksi SVR
        svr_pred_step = svr_model.predict(X_svr_scaled)[0]
        svr_preds.append(svr_pred_step)
        
        # 3. Hitung Prediksi Hybrid (Bobot 0.5/0.5 dari notebook)
        hybrid_pred_step = 0.5 * arima_preds.iloc[i] + 0.5 * svr_pred_step
        hybrid_preds.append(hybrid_pred_step)
        
        # 4. PENTING: Update lag untuk iterasi berikutnya
        current_lag_volume = hybrid_pred_step
        
    # Kembalikan hasil
    results_df = future_inputs_df.copy()
    results_df['Tahun'] = results_df['Tahun'].dt.year
    results_df['ARIMA_Pred'] = arima_preds.values
    results_df['SVR_Pred'] = svr_preds
    results_df['Prediksi_Hybrid (Volume Ribu Ekor)'] = hybrid_preds
    
    return results_df

# --- 2. SETUP TAMPILAN STREAMLIT ---

st.set_page_config(page_title="Prediksi Volume Ikan", layout="wide")
st.title("🐟 Aplikasi Prediksi Volume Produksi Ikan (Hybrid ARIMA-SVR)")

# Muat data
models_dict, data_hist = load_data()
fish_types = list(models_dict.keys())

# --- 3. UI (USER INTERFACE) ---

st.sidebar.header("Panel Kontrol")
selected_fish = st.sidebar.selectbox("Pilih Kelompok Ikan:", fish_types)
num_periods = st.sidebar.number_input("Jumlah Tahun Prediksi:", min_value=1, max_value=10, value=3)

st.header(f"Prediksi untuk: {selected_fish}")
st.write(f"""
Model ini membutuhkan asumsi Anda untuk **{num_periods} tahun ke depan**.
Silakan isi tabel di bawah dengan perkiraan nilai 'Nilai (Rp. Juta)' dan 'Harga Rata-Rata' untuk memprediksi 'Volume (Ribu Ekor)'.
""")

# Dapatkan tahun terakhir dari data
last_year = models_dict[selected_fish]['last_known_year'].year
future_years = [pd.to_datetime(f"{last_year + i + 1}-01-01") for i in range(num_periods)]

# Buat DataFrame kosong untuk diisi pengguna
df_input = pd.DataFrame({
    'Tahun': future_years,
    'Nilai (Rp. Juta)': [0.0] * num_periods,
    'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': [0.0] * num_periods
})

# Gunakan st.data_editor agar pengguna bisa mengedit nilai
st.subheader("Masukkan Asumsi Fitur Masa Depan:")
edited_df = st.data_editor(
    df_input,
    num_rows="dynamic",
    column_config={
        "Tahun": st.column_config.DateColumn(
            "Tahun Prediksi",
            format="YYYY",
            disabled=True
        ),
        "Nilai (Rp. Juta)": st.column_config.NumberColumn(
            "Nilai (Rp. Juta)",
            format="%.2f",
            required=True
        ),
        "Harga Rata-Rata Tertimbang(Rp/ ribu ekor)": st.column_config.NumberColumn(
            "Harga Rata-Rata (Rp/ribu ekor)",
            format="%.2f",
            required=True
        )
    },
    use_container_width=True,
    key=f"editor_{selected_fish}"
)

# Tombol untuk menjalankan prediksi
if st.button("Jalankan Prediksi", type="primary"):
    # Validasi input
    if edited_df['Nilai (Rp. Juta)'].eq(0).any() or edited_df['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'].eq(0).any():
        st.warning("Peringatan: Ada nilai 0 di input. Ini mungkin akan mempengaruhi hasil prediksi.")
    
    if edited_df.isnull().values.any():
        st.error("Error: Pastikan semua kolom input terisi.")
    else:
        with st.spinner("Menghitung prediksi rekursif..."):
            try:
                # Ambil komponen model yang relevan
                model_components = models_dict[selected_fish]
                
                # Panggil fungsi prediksi
                results_df = run_recursive_prediction(model_components, edited_df)
                
                st.subheader("Hasil Prediksi")
                display_cols = ['Tahun', 'ARIMA_Pred', 'SVR_Pred', 'Prediksi_Hybrid (Volume Ribu Ekor)']
                st.dataframe(results_df[display_cols].style.format({
                    'ARIMA_Pred': '{:,.2f}',
                    'SVR_Pred': '{:,.2f}',
                    'Prediksi_Hybrid (Volume Ribu Ekor)': '{:,.2f}'
                }))
                
                st.subheader("Grafik Prediksi Volume (Hybrid)")
                
                # Siapkan data untuk grafik
                chart_data = results_df.set_index('Tahun')['Prediksi_Hybrid (Volume Ribu Ekor)']
                
                st.line_chart(chart_data)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Historis (Agregat)")
st.sidebar.write(f"Menampilkan data historis untuk {selected_fish}:")
hist_display = data_hist[data_hist['Kelompok Ikan'] == selected_fish][['Volume (Ribu Ekor)', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']]
st.sidebar.dataframe(hist_display.style.format('{:,.2f}'))
