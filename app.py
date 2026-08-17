import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman (Harus dipanggil pertama kali)
st.set_page_config(
    page_title="Sistem Prediksi Segmentasi Pelanggan",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan CSS Kustom untuk mempercantik UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #FF4B4B;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .segment-card {
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat Model dan Scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model_artifacts/model_klasifikasi_dt.pkl')
        scaler = joblib.load('model_artifacts/scaler_klasifikasi.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_models()

# Bagian Header Utama
st.title("Sistem Prediksi Segmentasi Pelanggan")
st.divider()

# Mengecek apakah file pkl berhasil dibaca
if model is None or scaler is None:
    st.error("⚠️ ERROR: File model AI tidak ditemukan! Pastikan file **'model_klasifikasi_dt.pkl'** dan **'scaler_klasifikasi.pkl'** berada di dalam folder **'model_artifacts'**.")
    st.stop()


# ==========================================
# PREDIKSI MASAL DARI DATA MENTAH
# ==========================================
st.markdown("### 📁 Upload Data Transaksi")
st.info("Unggah file transaksi anda. Sistem mendukung file berformat **CSV** maupun **Excel**. Sistem akan **otomatis menghitung** nilai RFM untuk tiap pelanggan dan langsung memprediksi segmen mereka.")

uploaded_file = st.file_uploader("Pilih file dataset transaksi", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # 1. Membaca data mentah berdasarkan tipe file
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Berhasil membaca {len(df_raw)} baris data transaksi dari {uploaded_file.name}!")
        
        # 2. Cek dan Hitung TotalPrice jika belum ada
        if 'TotalPrice' not in df_raw.columns:
            # Periksa apakah ada Quantity dan UnitPrice sebagai gantinya
            if 'Quantity' in df_raw.columns and 'UnitPrice' in df_raw.columns:
                df_raw['TotalPrice'] = df_raw['Quantity'] * df_raw['UnitPrice']
        
        # 3. Validasi kolom wajib ada
        required_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        
        if missing_cols:
            st.error(f"⚠️ ERROR: File CSV Anda kehilangan kolom: **{', '.join(missing_cols)}**")
            st.write("Pastikan file memiliki: `CustomerID`, `InvoiceNo`, `InvoiceDate`. Jika `TotalPrice` tidak ada, wajib ada `Quantity` dan `UnitPrice`.")
        else:
            with st.spinner('⏳ Sedang mengekstrak fitur RFM dan memprediksi dengan AI...'):
                # 3. PROSES HITUNG OTOMATIS RFM
                # Mengatur agar mendukung format tanggal Indonesia (Hari/Bulan/Tahun)
                df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'], dayfirst=True, format='mixed')
                snapshot_date = df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)
                
                # Hitung Recency
                rfm_r = df_raw.groupby('CustomerID')['InvoiceDate'].max().reset_index()
                rfm_r['Recency'] = (snapshot_date - rfm_r['InvoiceDate']).dt.days
                
                # Hitung Frequency
                rfm_f = df_raw.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
                rfm_f.columns = ['CustomerID', 'Frequency']
                
                # Hitung Monetary
                rfm_m = df_raw.groupby('CustomerID')['TotalPrice'].sum().reset_index()
                rfm_m.columns = ['CustomerID', 'Monetary']
                
                # Gabungkan menjadi satu DataFrame RFM
                rfm_final = rfm_r[['CustomerID', 'Recency']].merge(rfm_f, on='CustomerID').merge(rfm_m, on='CustomerID')
                
                # 4. PREDIKSI SEGMENTASI
                X_pred = rfm_final[['Recency', 'Frequency', 'Monetary']]
                X_scaled = scaler.transform(X_pred)
                rfm_final['Prediksi_Segmen'] = model.predict(X_scaled)
                
            st.markdown("### 📊 Hasil Prediksi Seluruh Pelanggan")
            
            # Menampilkan Ringkasan Distribusi
            ringkasan = rfm_final['Prediksi_Segmen'].value_counts().reset_index()
            ringkasan.columns = ['Nama Segmen', 'Jumlah Pelanggan']
            
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.write("**Ringkasan Segmen:**")
                st.dataframe(ringkasan, use_container_width=True)
            with col_res2:
                st.write("**Detail Data Per Pelanggan:**")
                st.dataframe(rfm_final, use_container_width=True)
            


    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")


st.divider()
st.caption("Hak Cipta © 2026 | Sistem Analisis Segmentasi Pelanggan Berbasis Machine Learning")