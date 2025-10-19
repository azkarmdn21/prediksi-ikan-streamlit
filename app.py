import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import openpyxl 
import plotly.express as px # Import Plotly untuk visualisasi interaktif

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
        # Muat data historis (menggunakan read_excel)
        df_hist = pd.read_excel('produksi_pembenihan_jawaBarat_2019_2023.xlsx') 
        df_hist['Tahun'] = pd.to_datetime(df_hist['Tahun'], format='%Y')
        
        # Agregasi data yang sama seperti saat pelatihan
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
    """Menjalankan prediksi hybrid secara rekursif."""
    
    arima_result = model_components['arima_result']
    svr_model = model_components['svr_model']
    scaler = model_components['scaler']
    current_lag_volume = model_components['last_known_volume']
    n_periods = len(future_inputs_df)
    arima_preds = arima_result.forecast(steps=n_periods)
    
    svr_preds = []
    hybrid_preds = []
    
    for i in range(n_periods):
        future_nilai = future_inputs_df.iloc[i]['Nilai (Rp. Juta)']
        future_harga = future_inputs_df.iloc[i]['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']
        
        X_svr_future = pd.DataFrame({
            'Nilai (Rp. Juta)': [future_nilai],
            'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': [future_harga],
            'Volume_Lag1': [current_lag_volume]
        })
        
        X_svr_scaled = scaler.transform(X_svr_future)
        svr_pred_step = svr_model.predict(X_svr_scaled)[0]
        svr_preds.append(svr_pred_step)
        
        hybrid_pred_step = 0.5 * arima_preds.iloc[i] + 0.5 * svr_pred_step
        hybrid_preds.append(hybrid_pred_step)
        
        # Update lag rekursif
        current_lag_volume = hybrid_pred_step
        
    results_df = future_inputs_df.copy()
    results_df['Tahun'] = results_df['Tahun'].dt.year
    results_df['ARIMA_Pred'] = arima_preds.values.round(2)
    results_df['SVR_Pred'] = np.array(svr_preds).round(2)
    results_df['Prediksi_Hybrid (Volume Ribu Ekor)'] = np.array(hybrid_preds).round(2)
    
    return results_df

# --- 2. SETUP TAMPILAN STREAMLIT ---

st.set_page_config(page_title="Prediksi Volume Ikan Jabar", layout="wide")
st.title("🐟 Aplikasi Prediksi Volume Produksi Ikan (Hybrid ARIMA-SVR)")
st.caption("Dikembangkan dengan Streamlit dan Model Hybrid ARIMA-SVR")

# Muat data
models_dict, data_hist = load_data()
fish_types = sorted(list(models_dict.keys()))

# --- 3. UI (USER INTERFACE) ---

st.sidebar.header("Panel Kontrol & Input")
selected_fish = st.sidebar.selectbox("Pilih Kelompok Ikan:", fish_types)

# Dapatkan tahun terakhir dari data
last_year = models_dict[selected_fish]['last_known_year'].year
max_years = 10
num_periods = st.sidebar.slider("Jumlah Tahun Prediksi:", min_value=1, max_value=max_years, value=3)

# --- TAMPILAN GRAFIK HISTORIS (Default) ---
st.subheader(f"Tren Volume Produksi Historis {selected_fish}")
hist_data_volume = data_hist[data_hist['Kelompok Ikan'] == selected_fish]['Volume (Ribu Ekor)']
st.line_chart(hist_data_volume)

st.header(f"Form Input Asumsi ({selected_fish})")
st.info(f"""
Model membutuhkan input asumsi **Nilai (Rp. Juta)** dan **Harga Rata-Rata Tertimbang** untuk memprediksi Volume produksi dalam **{num_periods} tahun** ke depan.
""")

# Buat DataFrame kosong untuk diisi pengguna
future_years = [pd.to_datetime(f"{last_year + i + 1}-01-01") for i in range(num_periods)]
df_input = pd.DataFrame({
    'Tahun': future_years,
    'Nilai (Rp. Juta)': [hist_data_volume.iloc[-1] / 2] * num_periods, # Default diisi nilai estimasi awal
    'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': [hist_data_volume.iloc[-1] / 3] * num_periods
})

# Gunakan st.data_editor agar pengguna bisa mengedit nilai
st.subheader("Masukkan Asumsi Fitur Masa Depan:")
edited_df = st.data_editor(
    df_input,
    num_rows="fixed", # Membatasi baris sesuai input slider
    column_config={
        "Tahun": st.column_config.DateColumn("Tahun Prediksi", format="YYYY", disabled=True),
        "Nilai (Rp. Juta)": st.column_config.NumberColumn("Nilai (Rp. Juta)", format="%.2f", required=True),
        "Harga Rata-Rata Tertimbang(Rp/ ribu ekor)": st.column_config.NumberColumn("Harga Rata-Rata (Rp/ribu ekor)", format="%.2f", required=True)
    },
    use_container_width=True,
    key=f"editor_{selected_fish}_{num_periods}" 
)

# Tombol untuk menjalankan prediksi
if st.button("Jalankan Prediksi dan Tampilkan Hasil", type="primary"):
    
    # Validasi input sederhana
    if edited_df.isnull().values.any() or (edited_df[['Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']] == 0).any().any():
        st.error("Error: Pastikan semua kolom input terisi dan tidak bernilai nol (0).")
    else:
        with st.spinner(f"Menghitung prediksi rekursif {selected_fish} untuk {num_periods} tahun..."):
            try:
                model_components = models_dict[selected_fish]
                results_df = run_recursive_prediction(model_components, edited_df)
                
                st.markdown("---")
                st.success("✅ Prediksi Selesai! Lihat hasilnya di bawah.")
                
                # --- FITUR LAYOUT DUA KOLOM ---
                col_chart, col_table = st.columns([3, 2])
                
                with col_chart:
                    # --- FITUR PLOTLY GABUNGAN ---
                    st.subheader("Visualisasi Tren Historis & Prediksi")
                    
                    # 1. Siapkan DataFrame Historis
                    df_hist_volume = data_hist[data_hist['Kelompok Ikan'] == selected_fish].reset_index()
                    df_hist_volume['Tahun'] = df_hist_volume['Tahun'].dt.year
                    df_hist_volume = df_hist_volume[['Tahun', 'Volume (Ribu Ekor)']]
                    df_hist_volume['Jenis Data'] = 'Historis (Aktual)'
                    
                    # 2. Siapkan DataFrame Prediksi
                    df_pred_volume = results_df[['Tahun', 'Prediksi_Hybrid (Volume Ribu Ekor)']]
                    df_pred_volume.columns = ['Tahun', 'Volume (Ribu Ekor)']
                    df_pred_volume['Jenis Data'] = 'Prediksi (Hybrid)'

                    combined_df = pd.concat([df_hist_volume, df_pred_volume])
                    
                    fig = px.line(
                        combined_df, 
                        x='Tahun', 
                        y='Volume (Ribu Ekor)', 
                        color='Jenis Data', 
                        title=f"Volume {selected_fish} (2019 - {results_df['Tahun'].max()})",
                        markers=True
                    )
                    fig.update_layout(xaxis=dict(tickformat='d')) # Pastikan tahun ditampilkan sebagai integer
                    st.plotly_chart(fig, use_container_width=True)

                with col_table:
                    # --- TABEL HASIL PREDIIKSI ---
                    st.subheader("Tabel Hasil Prediksi")
                    display_cols = ['Tahun', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)', 'Prediksi_Hybrid (Volume Ribu Ekor)']
                    st.dataframe(
                        results_df[display_cols].style.format({
                            'Nilai (Rp. Juta)': '{:,.2f}',
                            'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': '{:,.2f}',
                            'Prediksi_Hybrid (Volume Ribu Ekor)': '{:,.2f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # --- FITUR DOWNLOAD HASIL ---
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv,
                        file_name=f'prediksi_{selected_fish}_{last_year+1}_to_{results_df["Tahun"].max()}.csv',
                        mime='text/csv',
                        type="secondary"
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- 4. FITUR INFORMASI TAMBAHAN (st.expander) ---
st.markdown("---")
with st.expander("ℹ️ Penjelasan Metodologi Model Hybrid ARIMA-SVR"):
    st.markdown("""
    Model yang digunakan dalam aplikasi ini adalah **Model Hybrid** yang dirancang untuk menggabungkan kekuatan dari dua jenis model prediksi:
    
    1.  **ARIMA (AutoRegressive Integrated Moving Average):**
        * Fokus pada analisis dan prediksi deret waktu berdasarkan pola internal data historis (**Volume**).
        * Digunakan untuk menangkap tren jangka panjang dan pola musiman yang konsisten.

    2.  **SVR (Support Vector Regression):**
        * Fokus pada pemodelan hubungan antara **Volume** sebagai variabel dependen dengan variabel independen (**Nilai** dan **Harga Rata-Rata**).
        * **Penting:** Model SVR ini bersifat **rekursif**, yang berarti nilai **Volume** prediksi dari periode sebelumnya (`Volume_Lag1`) digunakan sebagai salah satu fitur input untuk memprediksi volume pada periode saat ini. Ini yang membuat input asumsi Anda sangat krusial.

    **Prediksi Akhir (Hybrid)** didapatkan dengan rata-rata tertimbang (50% dari hasil ARIMA dan 50% dari hasil SVR).
    """)

# --- TAMPILAN HISTORIS DI SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("Data Historis (Agregat)")
st.sidebar.write(f"Data Agregat {selected_fish} (2019-{last_year}):")
hist_display = data_hist[data_hist['Kelompok Ikan'] == selected_fish][['Volume (Ribu Ekor)', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']]
st.sidebar.dataframe(hist_display.style.format('{:,.2f}'), height=200)
