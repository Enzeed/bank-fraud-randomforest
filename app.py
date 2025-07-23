import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("explore_randomforest_classification_2.pkl")

st.title("Prediksi Deteksi Transaksi Penipuan")

# Input dari user
transaction_amount = st.number_input("Jumlah Transaksi (TransactionAmount)", min_value=0.0)
transaction_type = st.selectbox("Tipe Transaksi (TransactionType)", ["Transfer", "Pembayaran", "Penarikan", "Setoran"])
location = st.selectbox("Lokasi Transaksi (Location)", ["Online", "Offline"])
channel = st.selectbox("Channel Transaksi (Channel)", ["Mobile", "ATM", "Internet Banking"])
customer_age = st.number_input("Umur Pelanggan (CustomerAge)", min_value=0)
customer_occupation = st.selectbox("Pekerjaan Pelanggan (CustomerOccupation)", ["Karyawan", "Wiraswasta", "Pelajar", "Pensiunan"])
transaction_duration = st.number_input("Durasi Transaksi (TransactionDuration)", min_value=0.0)
login_attempts = st.number_input("Jumlah Login Gagal (LoginAttempts)", min_value=0)
account_balance = st.number_input("Saldo Akun (AccountBalance)", min_value=0.0)
amount_scale = st.selectbox("Skala Jumlah Transaksi (AmountScale)", ["Low", "Medium", "High"])
balance_scale = st.selectbox("Skala Saldo Akun (AccountBalanceScale)", ["Low", "Medium", "High"])

# Ubah input kategorikal ke bentuk numerik (pastikan ini konsisten dengan preprocessing saat training)
# Kamu bisa pakai OneHotEncoder, LabelEncoder, atau mapping manual sesuai yang digunakan saat training
# Contoh mapping manual (ubah sesuai mapping asli):
transaction_type_map = {"Transfer": 0, "Pembayaran": 1, "Penarikan": 2, "Setoran": 3}
location_map = {"Online": 0, "Offline": 1}
channel_map = {"Mobile": 0, "ATM": 1, "Internet Banking": 2}
occupation_map = {"Karyawan": 0, "Wiraswasta": 1, "Pelajar": 2, "Pensiunan": 3}
scale_map = {"Low": 0, "Medium": 1, "High": 2}

# Buat array fitur
fitur = np.array([[
    transaction_amount,
    transaction_type_map[transaction_type],
    location_map[location],
    channel_map[channel],
    customer_age,
    occupation_map[customer_occupation],
    transaction_duration,
    login_attempts,
    account_balance,
    scale_map[amount_scale],
    scale_map[balance_scale]
]])

# Prediksi
if st.button("Prediksi"):
    hasil = model.predict(fitur)[0]
    if hasil == 1:
        st.error("⚠️ Transaksi Terindikasi Penipuan")
    else:
        st.success("✅ Transaksi Aman")
