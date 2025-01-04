import streamlit as st
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt

def get_today_date():
    """
    Bu fonksiyon, bugünün tarihini döndüren fonksiyondur.
    """
    today = datetime.datetime.today()
    return today.strftime('%Y-%m-%d'), today

def get_stock_data(symbol, start_date, end_date):
    """
    Bu fonksiyon, verilen hisse senedi sembolü için tarih aralığında veri çeker.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

def make_prediction(model, scaler, symbol, start_date="2023-01-01", end_date="2023-12-01"):
    """
    Model ile tahmin yapmak için bu fonksiyon kullanılır.
    """
    end_date_str, end_date_obj = get_today_date()
    
    data = get_stock_data(symbol, start_date, end_date_str)
    
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    recent_data = scaled_data[-28:]
    X_input = recent_data.reshape(1, 28, 1)
    
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    return predicted_price[0][0], end_date_obj, data

def main():
    st.title("Hisse Senedi Tahmin Uygulaması")

    symbol = st.text_input("Tahmin yapmak istediğiniz hisse senedi sembolünü girin:")
    
    if symbol:
        model = load_model('stock_prediction_model.h5')

        scaler = MinMaxScaler(feature_range=(0, 1))

        predicted_price, prediction_date, data = make_prediction(model, scaler, symbol)

        st.write(f"{symbol} için {prediction_date.strftime('%Y-%m-%d')} tarihinde tahmin edilen kapanış fiyatı: {predicted_price:.2f} USD")

        st.subheader(f"{symbol} - Son 28 Günün Kapanış Fiyatları")
        recent_days_data = data[-28:]
        plt.figure(figsize=(10,5))
        plt.plot(recent_days_data.index, recent_days_data.values, marker='o', label='Kapanış Fiyatı')
        plt.title(f"{symbol} Kapanış Fiyatları - Son 28 Gün")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)
        
        write_days_data = data[-7:]
        st.write(f"Son 7 Günün Kapanış Fiyatları:")
        for date, price in zip(write_days_data.index, write_days_data.values):
            st.write(f"{date.strftime('%Y-%m-%d')}: {price[0]:.2f} USD")


if __name__ == "__main__":
    main()
