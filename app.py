import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import openpyxl 
import plotly.express as px

warnings.filterwarnings('ignore')

# --- CONFIG APLIKASI (UI Enhancement) ---
st.set_page_config(
    page_title="Prediksi Ikan Jabar - Hybrid ARIMA-SVR", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🐟"
)

# --- 1. FUNGSI PEMUATAN DAN PREDIKSI ---

# Hapus fungsi ARIMA dan SVR dari file ini agar Streamlit tidak mengimpor ulang.
# Asumsikan semua dependensi model sudah dihandle oleh joblib saat model disimpan.
# Namun, karena model ARIMA membutuhkan kelas ARIMA, kita tetap pertahankan import di atas.

@st.cache_data
def load_data():
    """Memuat data historis dan model yang telah dilatih."""
    try:
        models_dict = joblib.load('all_models_hybrid.pkl')
    except FileNotFoundError:
        st.error("File model 'all_models_hybrid.pkl' tidak ditemukan. Pastikan file ada di folder yang sama.")
        st.stop()
        
    try:
        df_hist = pd.read_excel('produksi_pembenihan_jawaBarat_2019_2023.xlsx') 
        df_hist['Tahun'] = pd.to_datetime(df_hist['Tahun'], format='%Y')
        
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
        
        current_lag_volume = hybrid_pred_step
        
    results_df = future_inputs_df.copy()
    results_df['Tahun'] = results_df['Tahun'].dt.year
    results_df['ARIMA_Pred'] = arima_preds.values.round(2)
    results_df['SVR_Pred'] = np.array(svr_preds).round(2)
    results_df['Prediksi_Hybrid (Ribu Ekor)'] = np.array(hybrid_preds).round(2)
    
    return results_df

# --- 2. LAYOUT UTAMA ---

models_dict, data_hist = load_data()
fish_types = sorted(list(models_dict.keys()))

st.title("🐟 Proyeksi Produksi Benih Ikan Jawa Barat")
st.markdown("Aplikasi berbasis Model Hybrid ARIMA-SVR untuk memproyeksikan Volume produksi per jenis ikan.")

# --- SIDEBAR: KONTROL ---

st.sidebar.header("⚙️ Kontrol & Seleksi")
selected_fish = st.sidebar.selectbox("Pilih Kelompok Ikan:", fish_types)

last_year_data = models_dict[selected_fish]['last_known_year'].year
max_years = 10
num_periods = st.sidebar.slider("Jumlah Tahun Proyeksi:", min_value=1, max_value=max_years, value=3)

# Dapatkan data historis ikan terpilih
hist_data = data_hist[data_hist['Kelompok Ikan'] == selected_fish]
hist_last_row = hist_data.iloc[-1]
hist_last_volume = hist_last_row['Volume (Ribu Ekor)']
hist_last_nilai = hist_last_row['Nilai (Rp. Juta)']

# --- METRIC CARDS (Header Section) ---

st.markdown("---")
st.subheader(f"Dashboard Metrik Kunci untuk **{selected_fish}**")

# Hitung Rata-rata Volume Historis untuk Metrik Peningkatan
hist_avg_volume = hist_data['Volume (Ribu Ekor)'].mean()
volume_delta = hist_last_volume - hist_avg_volume

col_metric1, col_metric2, col_metric3 = st.columns(3)

with col_metric1:
    st.metric(
        label=f"Volume Aktual Terakhir ({last_year_data})",
        value=f"{hist_last_volume:,.0f} Ribu Ekor",
        delta=f"{volume_delta:,.0f} Ribu (vs. Rata-rata Hist.)",
        delta_color="normal"
    )

with col_metric2:
    st.metric(
        label="Rata-rata Nilai (2019 - 2023)",
        value=f"Rp {hist_data['Nilai (Rp. Juta)'].mean():,.2f} Juta"
    )

with col_metric3:
    st.metric(
        label="Rata-rata Harga (2019 - 2023)",
        value=f"Rp {hist_data['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)'].mean():,.0f}"
    )
    
st.markdown("---")

# --- 3. FORM INPUT ASUMSI ---

st.subheader(f"📝 Asumsi Input Ekonomi ({last_year_data + 1} - {last_year_data + num_periods})")
st.info(f"""
Model membutuhkan input asumsi **Nilai (Rp. Juta)** dan **Harga Rata-Rata** untuk {num_periods} tahun ke depan. Nilai default diisi dengan data aktual terakhir ({last_year_data}).
""")

future_years = [pd.to_datetime(f"{last_year_data + i + 1}-01-01") for i in range(num_periods)]

# Nilai default diisi dengan nilai aktual terakhir
default_nilai = [hist_last_nilai] * num_periods
default_harga = [hist_last_row['Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']] * num_periods

df_input = pd.DataFrame({
    'Tahun': future_years,
    'Nilai (Rp. Juta)': default_nilai, 
    'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': default_harga
})

edited_df = st.data_editor(
    df_input,
    num_rows="fixed",
    column_config={
        "Tahun": st.column_config.DateColumn("Tahun Proyeksi", format="YYYY", disabled=True),
        "Nilai (Rp. Juta)": st.column_config.NumberColumn("Nilai Asumsi (Rp Juta)", format="%.2f", required=True),
        "Harga Rata-Rata Tertimbang(Rp/ ribu ekor)": st.column_config.NumberColumn("Harga Rata-Rata Asumsi (Rp/ribu ekor)", format="%.2f", required=True)
    },
    use_container_width=True,
    key=f"editor_{selected_fish}_{num_periods}" 
)

# --- 4. TOMBOL PREDIIKSI ---

if st.button("🚀 Jalankan Proyeksi", type="primary"):
    
    if edited_df.isnull().values.any() or (edited_df[['Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)']] == 0).any().any():
        st.error("Error: Pastikan semua kolom asumsi terisi dan tidak bernilai nol (0).")
    else:
        with st.spinner(f"Menghitung proyeksi rekursif {selected_fish}..."):
            try:
                model_components = models_dict[selected_fish]
                results_df = run_recursive_prediction(model_components, edited_df)
                
                # --- METRIK HASIL PREDIKSI ---
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                # Hitung Total Prediksi
                total_pred_volume = results_df['Prediksi_Hybrid (Ribu Ekor)'].sum()
                avg_pred_volume = results_df['Prediksi_Hybrid (Ribu Ekor)'].mean()
                delta_pred = avg_pred_volume - hist_last_volume # Peningkatan vs tahun terakhir
                
                with col_res1:
                    st.metric(
                        label="Total Volume Proyeksi",
                        value=f"{total_pred_volume:,.0f} Ribu Ekor",
                        help="Total Volume Prediksi Hybrid selama periode simulasi."
                    )
                with col_res2:
                    st.metric(
                        label="Rata-rata Volume Proyeksi",
                        value=f"{avg_pred_volume:,.0f} Ribu Ekor",
                        delta=f"{delta_pred:,.0f} Ribu (vs. Vol. {last_year_data})",
                        delta_color="normal"
                    )
                with col_res3:
                    st.metric(
                        label="Tahun Prediksi Terakhir",
                        value=f"{results_df['Tahun'].max()}",
                        help=f"Volume di tahun terakhir diprediksi sebesar {results_df['Prediksi_Hybrid (Ribu Ekor)'].iloc[-1]:,.0f} Ribu Ekor."
                    )
                    
                st.markdown("---")
                
                # --- GRAFIK DAN TABEL HASIL (TATA LETAK KOLOM) ---
                
                col_chart, col_table = st.columns([3, 2])
                
                with col_chart:
                    st.subheader("Visualisasi Tren Historis & Proyeksi")
                    
                    # 1. Siapkan DataFrame Historis
                    df_hist_volume = hist_data.reset_index()
                    df_hist_volume['Tahun'] = df_hist_volume['Tahun'].dt.year
                    df_hist_volume = df_hist_volume[['Tahun', 'Volume (Ribu Ekor)']]
                    df_hist_volume.columns = ['Tahun', 'Volume (Ribu Ekor)']
                    df_hist_volume['Jenis Data'] = 'Aktual (Historis)'
                    
                    # 2. Siapkan DataFrame Prediksi
                    df_pred_volume = results_df[['Tahun', 'Prediksi_Hybrid (Ribu Ekor)']]
                    df_pred_volume.columns = ['Tahun', 'Volume (Ribu Ekor)']
                    df_pred_volume['Jenis Data'] = 'Proyeksi (Hybrid)'

                    combined_df = pd.concat([df_hist_volume, df_pred_volume])
                    
                    fig = px.line(
                        combined_df, 
                        x='Tahun', 
                        y='Volume (Ribu Ekor)', 
                        color='Jenis Data', 
                        title=f"Tren Volume {selected_fish} (2019 - {results_df['Tahun'].max()})",
                        markers=True,
                        template="plotly_white" # Template yang lebih bersih
                    )
                    fig.update_layout(xaxis=dict(tickformat='d'))
                    st.plotly_chart(fig, use_container_width=True)

                with col_table:
                    st.subheader("Tabel Detail Hasil Proyeksi")
                    display_cols = ['Tahun', 'Nilai (Rp. Juta)', 'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)', 'Prediksi_Hybrid (Ribu Ekor)']
                    st.dataframe(
                        results_df[display_cols].style.format({
                            'Nilai (Rp. Juta)': 'Rp {:,.2f}',
                            'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': 'Rp {:,.0f}',
                            'Prediksi_Hybrid (Ribu Ekor)': '{:,.0f}'
                        }),
                        use_container_width=True,
                        height=500
                    )
                    
                    # --- FITUR DOWNLOAD HASIL ---
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Proyeksi (CSV)",
                        data=csv,
                        file_name=f'proyeksi_{selected_fish}_{last_year_data+1}_to_{results_df["Tahun"].max()}.csv',
                        mime='text/csv',
                        type="secondary"
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- 5. FITUR INFORMASI TAMBAHAN (st.expander) ---
st.markdown("---")
with st.expander("🔬 Detail Metodologi Hybrid ARIMA-SVR"):
    st.markdown("""
    Model ini menggunakan pendekatan **Hybrid Rekursif** (50% ARIMA + 50% SVR) yang menggabungkan:
    
    * **ARIMA (Pola Waktu):** Fokus pada tren intrinsik Volume historis.
    * **SVR (Eksogen & Rekursif):** Memodelkan dampak dari asumsi **Nilai** dan **Harga** serta menggunakan **Volume prediksi tahun sebelumnya** sebagai *input lag* untuk memprediksi Volume tahun berikutnya.
    
    Pendekatan ini menghasilkan proyeksi yang lebih stabil dan responsif terhadap skenario ekonomi yang Anda inputkan.
    """)

# --- SIDEBAR: DATA HISTORIS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Data Historis Terakhir")
st.sidebar.dataframe(
    hist_last_row.drop('Kelompok Ikan').to_frame().T.style.format({
        'Volume (Ribu Ekor)': '{:,.0f}',
        'Nilai (Rp. Juta)': 'Rp {:,.2f}',
        'Harga Rata-Rata Tertimbang(Rp/ ribu ekor)': 'Rp {:,.0f}'
    }), 
    use_container_width=True
)
