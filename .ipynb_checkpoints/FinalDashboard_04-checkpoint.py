import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Linearitas Pendidikan vs Pekerjaan",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('linearitas_final_clean.csv')
        
        # Buat kolom Pendidikan_Label jika belum ada
        if 'Pendidikan_Encoded' in df.columns:
            education_map = {
                0: 'Tidak Sekolah',
                1: 'SD',
                2: 'SMP',
                3: 'SMA/SMK',
                4: 'Diploma',
                5: 'Sarjana',
                6: 'Magister',
                7: 'Doktor'
            }
            df['Pendidikan_Label'] = df['Pendidikan_Encoded'].map(education_map)
        
        # Buat kolom Gaji_Kategori jika belum ada
        if 'Gaji_Saat_Ini_Encoded' in df.columns:
            salary_map = {
                0: 'Tidak Ada/Belum Bekerja',
                1: '< 1 Juta',
                2: '1-3 Juta',
                3: '3-5 Juta',
                4: '5-10 Juta',
                5: '10-15 Juta',
                6: '> 15 Juta'
            }
            df['Gaji_Kategori'] = df['Gaji_Saat_Ini_Encoded'].map(salary_map)
        
        # Buat kolom Pengalaman_Kerja jika belum ada
        if 'Lama_Kerja_Saat_Ini_Encoded' in df.columns:
            experience_map = {
                0: 'Tidak Ada',
                1: '< 1 Tahun',
                2: '1-3 Tahun',
                3: '3-5 Tahun',
                4: '> 5 Tahun'
            }
            df['Pengalaman_Kerja'] = df['Lama_Kerja_Saat_Ini_Encoded'].map(experience_map)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Fungsi untuk memuat model
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('model_rf_linearitas.pkl')
        nb_model = joblib.load('model_nb_linearitas.pkl')
        return rf_model, nb_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Fungsi untuk menghitung metrics model secara real-time
def calculate_model_performance(model, X_test, y_test, model_name="Model"):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'Model': model_name,
            'Akurasi': accuracy,
            'Presisi': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score']
        }
        return metrics, y_pred
    except Exception as e:
        st.error(f"Error calculating metrics for {model_name}: {e}")
        return None, None

# Fungsi untuk menghitung feature importance secara real
def calculate_feature_importance(model, X, y, feature_names, model_type='rf'):
    try:
        if model_type == 'rf' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'Faktor': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Map feature names to readable names
            feature_map = {
                'Pendidikan_Encoded': 'Jenjang Pendidikan',
                'Jurusan_Cleaned_Final_V3_Encoded_Label': 'Jurusan',
                'Jenis Kelamin_Encoded_Label': 'Gender',
                'Kabupaten/Kota domisili anda_Encoded_Label': 'Domisili',
                'Status Pernikahan anda_Encoded_Label': 'Pernikahan',
                'Gaji_Saat_Ini_Encoded': 'Tingkat Gaji',
                'Group_Pekerjaan_Encoded_Label': 'Bidang Pekerjaan'
            }
            feature_imp['Faktor'] = feature_imp['Faktor'].replace(feature_map)
            return feature_imp
        
        elif model_type == 'nb':
            # Untuk Naive Bayes, gunakan permutation importance
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importances = result.importances_mean
            
            feature_imp = pd.DataFrame({
                'Faktor': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Map feature names to readable names
            feature_map = {
                'Pendidikan_Encoded': 'Jenjang Pendidikan',
                'Jurusan_Cleaned_Final_V3_Encoded_Label': 'Jurusan',
                'Jenis Kelamin_Encoded_Label': 'Gender',
                'Kabupaten/Kota domisili anda_Encoded_Label': 'Domisili',
                'Status Pernikahan anda_Encoded_Label': 'Pernikahan'
            }
            feature_imp['Faktor'] = feature_imp['Faktor'].replace(feature_map)
            return feature_imp
            
    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
        return None

# Load data dan model
df = load_data()
rf_model, nb_model = load_models()

# Sidebar
with st.sidebar:
    
    st.title("Navigasi Dashboard")
    
    menu_options = [
        "Overview & Dashboard",
        "Profil Responden",
        "Analisis Pendidikan",
        "Analisis Pekerjaan",
        "Linearitas Pendidikan-Pekerjaan",
        "Prediksi Model ML",
        "Insights & Rekomendasi"
    ]
    
    selected_page = st.radio("Pilih Menu:", menu_options)
    
    st.markdown("---")
    
    if df is not None:
        st.info(f"Total Data: {len(df):,} responden")
        st.info(f"Lokasi: Sumatera Utara")
        
    st.markdown("---")
    st.markdown("### Tentang Proyek")
    st.markdown("""
    Dashboard ini menganalisis hubungan antara 
    pendidikan dengan pekerjaan di Sumatera Utara.
    
    Tujuan:
    - Memahami tingkat linearitas pendidikan-pekerjaan
    - Mengidentifikasi faktor penentu linearitas
    - Memberikan rekomendasi kebijakan
    """)

# Halaman utama
st.markdown('<h1 class="main-header">Analisis Linearitas Pendidikan vs Pekerjaan di Sumatera Utara</h1>', unsafe_allow_html=True)

# 1. OVERVIEW & DASHBOARD
if selected_page == "Overview & Dashboard":
    st.markdown('<h2 class="sub-header">Dashboard Ringkasan</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # Hitung metrics secara real-time
        total_non_linear = len(df[df['Linearitas_System_Encoded'] == 0]) if 'Linearitas_System_Encoded' in df.columns else 0
        total_linear = len(df[df['Linearitas_System_Encoded'] == 1]) if 'Linearitas_System_Encoded' in df.columns else 0
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            percentage_non_linear = (total_non_linear / len(df) * 100) if len(df) > 0 else 0
            st.metric("Tidak Linear", f"{total_non_linear:,}", 
                     f"{percentage_non_linear:.1f}%")
        
        with col2:
            percentage_linear = (total_linear / len(df) * 100) if len(df) > 0 else 0
            st.metric("Linear", f"{total_linear:,}", 
                     f"{percentage_linear:.1f}%")
        
        with col3:
            if 'Pendidikan_Encoded' in df.columns:
                avg_education = df['Pendidikan_Encoded'].mean()
                st.metric("Rata-rata Pendidikan", f"{avg_education:.1f}", "skala 0-7")
            else:
                st.metric("Rata-rata Pendidikan", "N/A")
        
        with col4:
            if 'Jurusan_Cleaned_Final_V3' in df.columns:
                major_fields = df['Jurusan_Cleaned_Final_V3'].nunique()
                st.metric("Bidang Jurusan", f"{major_fields}")
            else:
                st.metric("Bidang Jurusan", "N/A")
        
        # Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Distribusi Linearitas</h3>', unsafe_allow_html=True)
            
            if 'Linearitas_System_Encoded' in df.columns:
                linear_counts = df['Linearitas_System_Encoded'].value_counts()
                
                fig = px.pie(values=linear_counts.values, 
                            names=['Tidak Linear', 'Linear'],
                            color=['Tidak Linear', 'Linear'],
                            color_discrete_map={'Tidak Linear':'#EF4444', 'Linear':'#10B981'},
                            hole=0.4)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False, 
                                height=400,
                                annotations=[dict(text=f'{len(df):,}Responden', 
                                                x=0.5, y=0.5, font_size=14, showarrow=False)])
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div class="info-box">
                <strong>Interpretasi:</strong> {percentage_non_linear:.1f}% responden bekerja 
                di luar bidang pendidikannya. Hal ini menunjukkan adanya mismatch 
                antara pendidikan dengan pekerjaan di Sumatera Utara.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h3 class="sub-header">Linearitas per Jenjang Pendidikan</h3>', unsafe_allow_html=True)
            
            if 'Pendidikan_Label' in df.columns and 'Linearitas_System_Encoded' in df.columns:
                # Hitung persentase real-time
                education_linearity = []
                for edu_level in df['Pendidikan_Label'].unique():
                    subset = df[df['Pendidikan_Label'] == edu_level]
                    total = len(subset)
                    linear = len(subset[subset['Linearitas_System_Encoded'] == 1])
                    non_linear = len(subset[subset['Linearitas_System_Encoded'] == 0])
                    
                    if total > 0:
                        education_linearity.append({
                            'Pendidikan': edu_level,
                            'Tidak Linear': (non_linear / total * 100),
                            'Linear': (linear / total * 100)
                        })
                
                edu_df = pd.DataFrame(education_linearity)
                
                # Urutkan sesuai jenjang
                order = ['Tidak Sekolah', 'SD', 'SMP', 'SMA/SMK', 'Diploma', 'Sarjana', 'Magister', 'Doktor']
                edu_df['Pendidikan'] = pd.Categorical(edu_df['Pendidikan'], categories=order, ordered=True)
                edu_df = edu_df.sort_values('Pendidikan')
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=edu_df['Pendidikan'],
                    x=edu_df['Tidak Linear'],
                    name='Tidak Linear',
                    orientation='h',
                    marker_color='#EF4444'
                ))
                
                fig.add_trace(go.Bar(
                    y=edu_df['Pendidikan'],
                    x=edu_df['Linear'],
                    name='Linear',
                    orientation='h',
                    marker_color='#10B981'
                ))
                
                fig.update_layout(
                    barmode='stack',
                    height=400,
                    title="Persentase Linearitas berdasarkan Jenjang Pendidikan",
                    xaxis_title="Persentase (%)",
                    yaxis_title="Jenjang Pendidikan",
                    legend_title="Status Linearitas"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Top Jurusan Tidak Linear (real-time calculation)
        st.markdown('<h3 class="sub-header">Top 10 Jurusan yang Paling Sering Bekerja TIDAK Linear</h3>', unsafe_allow_html=True)
        
        if 'Jurusan_Cleaned_Final_V3' in df.columns and 'Linearitas_System_Encoded' in df.columns:
            df_non_linear = df[df['Linearitas_System_Encoded'] == 0]
            
            if len(df_non_linear) > 0:
                top_mismatch = df_non_linear['Jurusan_Cleaned_Final_V3'].value_counts().head(10)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(y=top_mismatch.index,
                                x=top_mismatch.values,
                                orientation='h',
                                color=top_mismatch.values,
                                color_continuous_scale='reds',
                                text=top_mismatch.values)
                    
                    fig.update_layout(
                        height=400,
                        title="Jurusan dengan Tingkat Mismatch Tertinggi",
                        xaxis_title="Jumlah Responden",
                        yaxis_title="Jurusan",
                        showlegend=False
                    )
                    
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Analisis Cepat:")
                    for i, (jurusan, count) in enumerate(top_mismatch.items(), 1):
                        percentage = (count / total_non_linear * 100) if total_non_linear > 0 else 0
                        st.metric(f"{i}. {jurusan[:20]}...", 
                                 f"{count}", 
                                 f"{percentage:.1f}%")
            else:
                st.warning("Tidak ada data tidak linear untuk ditampilkan")

# 6. PREDIKSI MODEL ML (REAL-TIME CALCULATIONS)
elif selected_page == "Prediksi Model ML":
    st.markdown('<h2 class="sub-header">Analisis Model Machine Learning</h2>', unsafe_allow_html=True)
    
    if rf_model is not None and nb_model is not None and df is not None:
        # Siapkan data untuk evaluasi
        try:
            # Definisikan fitur yang digunakan dalam training
            fitur_cols = [
                'Pendidikan_Encoded',
                'Jurusan_Cleaned_Final_V3_Encoded_Label'
            ]
            
            # Pastikan semua fitur ada di dataframe
            available_features = [col for col in fitur_cols if col in df.columns]
            
            if len(available_features) > 0:
                X = df[available_features]
                y = df['Linearitas_System_Encoded']
                
                # Split data untuk evaluasi
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Hitung metrics untuk kedua model
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<h3 class="sub-header">Performa Model (Real-time)</h3>', unsafe_allow_html=True)
                    
                    # Random Forest metrics
                    rf_metrics, rf_pred = calculate_model_performance(rf_model, X_test, y_test, "Random Forest")
                    
                    # Naive Bayes metrics
                    nb_metrics, nb_pred = calculate_model_performance(nb_model, X_test, y_test, "Naive Bayes")
                    
                    if rf_metrics and nb_metrics:
                        # Buat DataFrame untuk visualisasi
                        metrics_data = [rf_metrics, nb_metrics]
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        fig = go.Figure(data=[
                            go.Bar(name='Random Forest',
                                  x=metrics_df.columns[1:],
                                  y=metrics_df.iloc[0, 1:],
                                  marker_color='#3B82F6'),
                            go.Bar(name='Naive Bayes',
                                  x=metrics_df.columns[1:],
                                  y=metrics_df.iloc[1, 1:],
                                  marker_color='#10B981')
                        ])
                        
                        fig.update_layout(barmode='group',
                                        height=400,
                                        title="Perbandingan Performa Model ML (Real Calculation)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan metrics detail
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Random Forest Accuracy", 
                                     f"{rf_metrics['Akurasi']:.2%}")
                            st.metric("Random Forest F1-Score", 
                                     f"{rf_metrics['F1-Score']:.2%}")
                        
                        with col_b:
                            st.metric("Naive Bayes Accuracy", 
                                     f"{nb_metrics['Akurasi']:.2%}")
                            st.metric("Naive Bayes F1-Score", 
                                     f"{nb_metrics['F1-Score']:.2%}")
                    
                with col2:
                    st.markdown('<h3 class="sub-header">Feature Importance (Real Calculation)</h3>', unsafe_allow_html=True)
                    
                    # Hitung feature importance untuk Random Forest
                    rf_importance = calculate_feature_importance(rf_model, X, y, available_features, 'rf')
                    
                    if rf_importance is not None:
                        fig = px.bar(rf_importance,
                                    x='Importance',
                                    y='Faktor',
                                    orientation='h',
                                    color='Importance',
                                    color_continuous_scale='viridis',
                                    text='Importance')
                        
                        fig.update_layout(height=400,
                                        title="Faktor Penentu Linearitas (Random Forest)",
                                        xaxis_title="Tingkat Pengaruh",
                                        yaxis_title="Faktor")
                        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan insight real
                        top_factor = rf_importance.iloc[-1]['Faktor'] if len(rf_importance) > 0 else "N/A"
                        top_value = rf_importance.iloc[-1]['Importance'] if len(rf_importance) > 0 else 0
                        st.markdown(f"### Insight Real: {top_factor} adalah faktor terpenting ({top_value:.1%})")
                    
                    # Tambahkan confusion matrix
                    st.markdown('<h3 class="sub-header">Confusion Matrix</h3>', unsafe_allow_html=True)
                    
                    if rf_pred is not None:
                        cm = confusion_matrix(y_test, rf_pred)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Prediksi: Tidak Linear', 'Prediksi: Linear'],
                            y=['Aktual: Tidak Linear', 'Aktual: Linear'],
                            colorscale='Blues',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 16}
                        ))
                        
                        fig.update_layout(height=300,
                                        title="Confusion Matrix - Random Forest",
                                        xaxis_title="Prediksi Model",
                                        yaxis_title="Data Aktual")
                        st.plotly_chart(fig, use_container_width=True)
                        
            else:
                st.warning("Fitur yang diperlukan tidak ditemukan dalam dataset")
                
        except Exception as e:
            st.error(f"Error dalam evaluasi model: {e}")
    
    else:
        st.warning("Model atau data tidak tersedia. Pastikan file model dan data sudah di-load dengan benar.")

# 2. PROFIL RESPONDEN
elif selected_page == "Profil Responden":
    st.markdown('<h2 class="sub-header">Profil Demografi Responden</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Distribusi Jenis Kelamin</h3>', unsafe_allow_html=True)
            
            if 'Jenis Kelamin' in df.columns:
                gender_counts = df['Jenis Kelamin'].value_counts()
                
                fig = px.pie(values=gender_counts.values,
                            names=gender_counts.index,
                            hole=0.3,
                            color_discrete_sequence=['#3B82F6', '#EC4899'])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div class="info-box">
                <strong>Komposisi:</strong><br>
                - Pria: {gender_counts.get('Pria', 0):,} ({gender_counts.get('Pria', 0)/len(df)*100:.1f}%)<br>
                - Wanita: {gender_counts.get('Wanita', 0):,} ({gender_counts.get('Wanita', 0)/len(df)*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Kolom 'Jenis Kelamin' tidak ditemukan")
        
        with col2:
            st.markdown('<h3 class="sub-header">Status Pernikahan</h3>', unsafe_allow_html=True)
            
            if 'Status Pernikahan anda' in df.columns:
                marriage_counts = df['Status Pernikahan anda'].value_counts()
                
                fig = px.bar(x=marriage_counts.values,
                            y=marriage_counts.index,
                            orientation='h',
                            color=marriage_counts.values,
                            color_continuous_scale='viridis')
                
                fig.update_layout(height=300,
                                xaxis_title="Jumlah Responden",
                                yaxis_title="Status")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Kolom 'Status Pernikahan anda' tidak ditemukan")
        
        # Distribusi Domisili
        st.markdown('<h3 class="sub-header">Distribusi Domisili Responden</h3>', unsafe_allow_html=True)
        
        if 'Kabupaten/Kota domisili anda ' in df.columns:
            # Filter untuk top 15 kota
            city_counts = df['Kabupaten/Kota domisili anda '].value_counts().head(15)
            
            fig = px.treemap(names=city_counts.index,
                            parents=['Sumatera Utara'] * len(city_counts),
                            values=city_counts.values,
                            color=city_counts.values,
                            color_continuous_scale='Blues')
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabel detail
            with st.expander("Lihat Detail Distribusi Kota/Kabupaten"):
                city_details = df['Kabupaten/Kota domisili anda '].value_counts().reset_index()
                city_details.columns = ['Kota/Kabupaten', 'Jumlah Responden']
                city_details['Persentase'] = (city_details['Jumlah Responden'] / len(df) * 100).round(2)
                st.dataframe(city_details, use_container_width=True)
        else:
            st.warning("Kolom 'Kabupaten/Kota domisili anda' tidak ditemukan")
        
        # Usia (jika ada kolom usia)
        if 'Berapa usia anda saat ini? (cth: 20 tahun)' in df.columns:
            st.markdown('<h3 class="sub-header">Distribusi Usia</h3>', unsafe_allow_html=True)
            
            # Clean age data
            df_clean = df.copy()
            df_clean['Usia'] = pd.to_numeric(df_clean['Berapa usia anda saat ini? (cth: 20 tahun)'], errors='coerce')
            df_clean = df_clean.dropna(subset=['Usia'])
            
            fig = px.histogram(df_clean, x='Usia', nbins=20,
                              title='Distribusi Usia Responden',
                              labels={'Usia': 'Usia (tahun)'})
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Usia Rata-rata", f"{df_clean['Usia'].mean():.1f} tahun")
            with col2:
                st.metric("Usia Minimum", f"{df_clean['Usia'].min():.0f} tahun")
            with col3:
                st.metric("Usia Maksimum", f"{df_clean['Usia'].max():.0f} tahun")

# 3. ANALISIS PENDIDIKAN - DIBAWAH BAGIAN HEATMAP (PERBAIKAN DATA NUMERIK)
elif selected_page == "Analisis Pendidikan":
    st.markdown('<h2 class="sub-header">Analisis Latar Belakang Pendidikan</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Distribusi Jenjang Pendidikan</h3>', unsafe_allow_html=True)
            
            if 'Pendidikan_Label' in df.columns:
                edu_counts = df['Pendidikan_Label'].value_counts()
                
                fig = px.bar(x=edu_counts.values,
                            y=edu_counts.index,
                            orientation='h',
                            color=edu_counts.values,
                            color_continuous_scale='blues',
                            text=edu_counts.values)
                
                fig.update_layout(height=400,
                                xaxis_title="Jumlah Responden",
                                yaxis_title="Jenjang Pendidikan")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data pendidikan label tidak tersedia")
        
        with col2:
            st.markdown('<h3 class="sub-header">Distribusi Bidang Jurusan (Top 15)</h3>', unsafe_allow_html=True)
            
            if 'Jurusan_Cleaned_Final_V3' in df.columns:
                major_counts = df['Jurusan_Cleaned_Final_V3'].value_counts().head(15)
                
                fig = px.bar(x=major_counts.values,
                            y=major_counts.index,
                            orientation='h',
                            color=major_counts.values,
                            color_continuous_scale='viridis',
                            text=major_counts.values)
                
                fig.update_layout(height=400,
                                xaxis_title="Jumlah Responden",
                                yaxis_title="Bidang Jurusan")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data jurusan tidak tersedia")
        
        # Heatmap: Pendidikan vs Jurusan - DATA REAL NUMERIK
        st.markdown('<h3 class="sub-header">Hubungan Pendidikan dengan Bidang Jurusan</h3>', unsafe_allow_html=True)
        
        if 'Pendidikan_Label' in df.columns and 'Jurusan_Cleaned_Final_V3' in df.columns:
            # ============================================================
            # BAGIAN 1: ANALISIS DATA REAL DENGAN NUMERIK BULAT
            # ============================================================
            
            # Buat cross tabulation dengan COUNT (bukan persentase)
            cross_tab_count = pd.crosstab(df['Pendidikan_Label'], 
                                         df['Jurusan_Cleaned_Final_V3'])
            
            # Urutkan jenjang pendidikan secara logis
            education_order = ['Tidak Sekolah', 'SD', 'SMP', 'SMA/SMK', 'Diploma', 'Sarjana', 'Magister', 'Doktor']
            
            # Filter hanya jenjang yang ada dalam data
            available_education = [edu for edu in education_order if edu in cross_tab_count.index]
            cross_tab_count = cross_tab_count.reindex(available_education)
            
            # Filter hanya jurusan yang memiliki data signifikan (minimal 5 total)
            total_by_jurusan = cross_tab_count.sum()
            significant_jurusan = total_by_jurusan[total_by_jurusan >= 5].index.tolist()
            cross_tab_count = cross_tab_count[significant_jurusan]
            
            # Batasi jumlah jurusan untuk visualisasi yang lebih baik
            if len(cross_tab_count.columns) > 15:
                # Ambil jurusan dengan total count tertinggi
                column_sums = cross_tab_count.sum().sort_values(ascending=False)
                top_columns = column_sums.head(15).index.tolist()
                cross_tab_count = cross_tab_count[top_columns]
            
            if len(cross_tab_count) > 0 and len(cross_tab_count.columns) > 0:
                # ============================================================
                # BAGIAN 2: HEATMAP DENGAN DATA NUMERIK BULAT
                # ============================================================
                
                # Buat heatmap dengan data COUNT (bukan persentase)
                fig = px.imshow(cross_tab_count,
                               text_auto=True,  # Tampilkan angka bulat
                               aspect="auto",
                               color_continuous_scale='viridis',
                               labels=dict(x="Bidang Jurusan", y="Jenjang Pendidikan", color="Jumlah Responden"),
                               title="Distribusi Jurusan per Jenjang Pendidikan (Jumlah Responden)")
                
                # Atur layout untuk readability
                fig.update_layout(
                    height=max(400, len(cross_tab_count) * 50),
                    width=max(800, len(cross_tab_count.columns) * 60),
                    xaxis_title="Bidang Jurusan",
                    yaxis_title="Jenjang Pendidikan",
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ============================================================
                # BAGIAN 3: TABEL DATA NUMERIK REAL (COUNT & PERCENTAGE)
                # ============================================================
                with st.expander("Lihat Data Numerik Heatmap (Data Real)"):
                    
                    # Tab untuk melihat data dalam format berbeda
                    tab1, tab2 = st.tabs(["Jumlah Responden (Count)", "Persentase per Baris"])
                    
                    with tab1:
                        st.markdown("### Tabel Data: Jumlah Responden per Pendidikan-Jurusan")
                        
                        # Tampilkan data count dengan styling
                        styled_count = cross_tab_count.style\
                            .background_gradient(cmap='viridis', axis=None)\
                            .format(precision=0)  # Format tanpa desimal
                        
                        st.dataframe(styled_count, use_container_width=True)
                        
                        # Ringkasan statistik
                        st.markdown("#### Ringkasan Statistik")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_responden = cross_tab_count.sum().sum()
                            st.metric("Total Responden dalam Heatmap", f"{total_responden:,}")
                        
                        with col2:
                            total_original = len(df)
                            coverage = (total_responden / total_original * 100) if total_original > 0 else 0
                            st.metric("Cakupan Data", f"{coverage:.1f}%")
                        
                        with col3:
                            unique_jurusan = len(cross_tab_count.columns)
                            st.metric("Jumlah Jurusan Ditampilkan", f"{unique_jurusan}")
                    
                    with tab2:
                        st.markdown("### Tabel Data: Persentase Distribusi per Jenjang Pendidikan")
                        
                        # Hitung persentase per baris (per jenjang)
                        cross_tab_percent = cross_tab_count.div(cross_tab_count.sum(axis=1), axis=0) * 100
                        
                        # Format ke integer (bulatkan ke angka bulat)
                        cross_tab_percent_int = cross_tab_percent.round(0).astype(int)
                        
                        # Tampilkan dengan styling
                        styled_percent = cross_tab_percent_int.style\
                            .background_gradient(cmap='RdYlGn', axis=1, vmin=0, vmax=100)\
                            .format(precision=0)\
                            .set_properties(**{'text-align': 'center'})
                        
                        st.dataframe(styled_percent, use_container_width=True)
                        
                        st.markdown("**Keterangan:** Angka menunjukkan persentase jurusan dalam setiap jenjang pendidikan")
                        
                        # Contoh interpretasi
                        st.markdown("#### Contoh Interpretasi:")
                        st.markdown("- **SD**: 100% IPA artinya semua responden SD mengambil jurusan IPA")
                        st.markdown("- **Sarjana**: Distribusi lebih beragam karena banyak pilihan jurusan")
                        
                    # ============================================================
                    # BAGIAN 4: ANALISIS DETAIL PER JENJANG
                    # ============================================================
                    st.markdown("---")
                    st.markdown("### Analisis Detail per Jenjang Pendidikan")
                    
                    # Pilih jenjang untuk analisis detail
                    selected_edu = st.selectbox(
                        "Pilih Jenjang Pendidikan untuk Analisis Detail:",
                        options=available_education,
                        index=min(3, len(available_education)-1) if available_education else 0
                    )
                    
                    if selected_edu:
                        # Data untuk jenjang terpilih
                        edu_data = cross_tab_count.loc[selected_edu]
                        edu_total = edu_data.sum()
                        
                        # Filter hanya jurusan dengan data > 0
                        edu_data_nonzero = edu_data[edu_data > 0]
                        
                        if len(edu_data_nonzero) > 0:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Buat bar chart untuk jenjang terpilih
                                fig = px.bar(
                                    x=edu_data_nonzero.values,
                                    y=edu_data_nonzero.index,
                                    orientation='h',
                                    color=edu_data_nonzero.values,
                                    color_continuous_scale='blues',
                                    text=edu_data_nonzero.values,
                                    title=f"Distribusi Jurusan: {selected_edu} (Total: {edu_total} responden)"
                                )
                                
                                fig.update_layout(
                                    height=300,
                                    xaxis_title="Jumlah Responden",
                                    yaxis_title="Jurusan"
                                )
                                fig.update_traces(textposition='outside')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Tampilkan jurusan utama
                                top_jurusan = edu_data_nonzero.idxmax()
                                top_count = edu_data_nonzero.max()
                                top_percentage = (top_count / edu_total * 100) if edu_total > 0 else 0
                                
                                st.metric("Jurusan Paling Populer", 
                                         top_jurusan,
                                         f"{top_count} responden")
                                
                                st.metric("Persentase", 
                                         f"{top_percentage:.0f}%")
                                
                                # Hitung keragaman
                                diversity = len(edu_data_nonzero)
                                st.metric("Jumlah Jurusan", f"{diversity}")
                        
                        # Tampilkan tabel detail
                        st.markdown(f"#### Data Detail untuk {selected_edu}")
                        
                        # Buat DataFrame detail
                        detail_df = pd.DataFrame({
                            'Jurusan': edu_data_nonzero.index,
                            'Jumlah Responden': edu_data_nonzero.values,
                            'Persentase': ((edu_data_nonzero.values / edu_total * 100) if edu_total > 0 else 0)
                        })
                        
                        # Format persentase ke integer
                        detail_df['Persentase'] = detail_df['Persentase'].round(0).astype(int)
                        detail_df = detail_df.sort_values('Jumlah Responden', ascending=False)
                        
                        # Tampilkan tabel
                        st.dataframe(
                            detail_df.style.format({
                                'Jumlah Responden': '{:,.0f}',
                                'Persentase': '{:,.0f}%'
                            }),
                            use_container_width=True
                        )
                
                # ============================================================
                # BAGIAN 5: INSIGHTS OTOMATIS DARI DATA
                # ============================================================
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### Informasi dari Data")
                
                # Hitung beberapa insights
                insights = []
                
                # 1. Jenjang dengan jurusan paling beragam
                diversity_by_edu = (cross_tab_count > 0).sum(axis=1)
                most_diverse_edu = diversity_by_edu.idxmax() if len(diversity_by_edu) > 0 else "N/A"
                most_diverse_count = diversity_by_edu.max() if len(diversity_by_edu) > 0 else 0
                insights.append(f"**{most_diverse_edu}** memiliki jurusan paling beragam ({most_diverse_count} jurusan)")
                
                # 2. Jurusan yang ada di banyak jenjang
                diversity_by_major = (cross_tab_count > 0).sum(axis=0)
                widespread_major = diversity_by_major.idxmax() if len(diversity_by_major) > 0 else "N/A"
                widespread_count = diversity_by_major.max() if len(diversity_by_major) > 0 else 0
                insights.append(f"**{widespread_major}** adalah jurusan yang tersedia di {widespread_count} jenjang berbeda")
                
                # Tampilkan insights
                for i, insight in enumerate(insights, 1):
                    st.markdown(f"{i}. {insight}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.warning("Tidak ada data yang memenuhi kriteria untuk heatmap")
                
                # Tampilkan data mentah sebagai alternatif
                with st.expander("Lihat Data Mentah Pendidikan-Jurusan"):
                    simple_cross = pd.crosstab(df['Pendidikan_Label'], df['Jurusan_Cleaned_Final_V3'])
                    st.dataframe(simple_cross, use_container_width=True)
        
        else:
            st.warning("Data tidak lengkap untuk analisis hubungan pendidikan-jurusan")
            
# 4. ANALISIS PEKERJAAN
elif selected_page == "Analisis Pekerjaan":
    st.markdown('<h2 class="sub-header">Profil Pekerjaan & Karir</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Status Pekerjaan Saat Ini</h3>', unsafe_allow_html=True)
            
            if 'Status_Pekerjaan_Final' in df.columns:
                job_status_counts = df['Status_Pekerjaan_Final'].value_counts().head(10)
                
                fig = px.bar(x=job_status_counts.values,
                            y=job_status_counts.index,
                            orientation='h',
                            color=job_status_counts.values,
                            color_continuous_scale='greens',
                            text=job_status_counts.values)
                
                fig.update_layout(height=400,
                                xaxis_title="Jumlah Responden",
                                yaxis_title="Status Pekerjaan")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Kolom 'Status_Pekerjaan_Final' tidak ditemukan")
        
        with col2:
            st.markdown('<h3 class="sub-header">Bidang Pekerjaan Saat Ini</h3>', unsafe_allow_html=True)
            
            if 'Group_Pekerjaan' in df.columns:
                field_counts = df['Group_Pekerjaan'].value_counts().head(10)
                
                fig = px.pie(values=field_counts.values,
                            names=field_counts.index,
                            hole=0.3,
                            color_discrete_sequence=px.colors.sequential.Viridis)
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Kolom 'Group_Pekerjaan' tidak ditemukan")
        
        # Analisis Gaji
        st.markdown('<h3 class="sub-header">Analisis Tingkat Penghasilan</h3>', unsafe_allow_html=True)
        
        if 'Gaji_Kategori' in df.columns:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                salary_counts = df['Gaji_Kategori'].value_counts()
                
                fig = px.bar(x=salary_counts.index,
                            y=salary_counts.values,
                            color=salary_counts.values,
                            color_continuous_scale='sunset',
                            text=salary_counts.values)
                
                fig.update_layout(height=400,
                                xaxis_title="Rentang Gaji",
                                yaxis_title="Jumlah Responden",
                                xaxis_tickangle=45)
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Statistik Gaji:")
                
                if 'Gaji_Saat_Ini_Encoded' in df.columns:
                    avg_salary = df['Gaji_Saat_Ini_Encoded'].mean()
                    median_salary = df['Gaji_Saat_Ini_Encoded'].median()
                    
                    st.metric("Rata-rata Level Gaji", f"{avg_salary:.1f}")
                    st.metric("Median Level Gaji", f"{median_salary:.1f}")
                    
                    # Persentase gaji > 3 Juta
                    high_earners = len(df[df['Gaji_Saat_Ini_Encoded'] >= 3])
                    st.metric("Penghasilan > 3 Juta", 
                             f"{high_earners:,}",
                             f"{(high_earners/len(df)*100):.1f}%")
        else:
            st.warning("Data gaji tidak tersedia")
        
        # Analisis Lama Kerja
        st.markdown('<h3 class="sub-header">Pengalaman Kerja</h3>', unsafe_allow_html=True)
        
        if 'Pengalaman_Kerja' in df.columns and 'Status_Pekerjaan_Final' in df.columns:
            fig = px.sunburst(df,
                             path=['Pengalaman_Kerja', 'Status_Pekerjaan_Final'],
                             title="Distribusi Pengalaman Kerja per Status Pekerjaan")
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Dashboard Interaktif: Filter berdasarkan pekerjaan
        st.markdown('<h3 class="sub-header">Filter Data Pekerjaan</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_job = st.selectbox(
                "Pilih Status Pekerjaan:",
                options=['Semua'] + list(df['Status_Pekerjaan_Final'].unique()) if 'Status_Pekerjaan_Final' in df.columns else ['Semua']
            )
        
        with col2:
            selected_field = st.selectbox(
                "Pilih Bidang Pekerjaan:",
                options=['Semua'] + list(df['Group_Pekerjaan'].unique()) if 'Group_Pekerjaan' in df.columns else ['Semua']
            )
        
        with col3:
            min_salary = st.select_slider(
                "Level Gaji Minimum:",
                options=[0, 1, 2, 3, 4, 5, 6],
                value=0,
                format_func=lambda x: {
                    0: 'Tidak Ada/Belum Bekerja',
                    1: '< 1 Juta',
                    2: '1-3 Juta',
                    3: '3-5 Juta',
                    4: '5-10 Juta',
                    5: '10-15 Juta',
                    6: '> 15 Juta'
                }[x]
            )
        
        # Filter data
        filtered_df = df.copy()
        
        # Tampilkan kolom yang tersedia
        available_cols = []
        display_cols = []
        
        if 'Status_Pekerjaan_Final' in filtered_df.columns:
            if selected_job != 'Semua':
                filtered_df = filtered_df[filtered_df['Status_Pekerjaan_Final'] == selected_job]
            available_cols.append('Status_Pekerjaan_Final')
            display_cols.append('Status_Pekerjaan_Final')
        
        if 'Group_Pekerjaan' in filtered_df.columns:
            if selected_field != 'Semua':
                filtered_df = filtered_df[filtered_df['Group_Pekerjaan'] == selected_field]
            available_cols.append('Group_Pekerjaan')
            display_cols.append('Group_Pekerjaan')
        
        if 'Gaji_Kategori' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Gaji_Saat_Ini_Encoded'] >= min_salary]
            available_cols.append('Gaji_Kategori')
            display_cols.append('Gaji_Kategori')
        
        if 'Linearitas_System_Encoded' in filtered_df.columns:
            available_cols.append('Linearitas_System_Encoded')
            display_cols.append('Linearitas_System_Encoded')
        
        # Tampilkan statistik
        st.metric("Jumlah Responden yang Sesuai Filter", f"{len(filtered_df):,}")
        
        if len(filtered_df) > 0 and 'Linearitas_System_Encoded' in filtered_df.columns:
            avg_linear = filtered_df['Linearitas_System_Encoded'].mean() * 100
            st.metric("Persentase Linearitas", f"{avg_linear:.1f}%")
            
            with st.expander("Lihat Data Terfilter"):
                st.dataframe(filtered_df[display_cols].head(20), use_container_width=True)
        elif len(filtered_df) > 0:
            with st.expander("Lihat Data Terfilter"):
                st.dataframe(filtered_df[display_cols].head(20), use_container_width=True)

# 5. LINEARITAS PENDIDIKAN-PEKERJAAN
elif selected_page == "Linearitas Pendidikan-Pekerjaan":
    st.markdown('<h2 class="sub-header">Analisis Linearitas Pendidikan vs Pekerjaan</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Tingkat Linearitas Keseluruhan</h3>', unsafe_allow_html=True)
            
            if 'Linearitas_System_Encoded' in df.columns:
                linear_stats = df['Linearitas_System_Encoded'].value_counts()
                
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = (linear_stats.get(1, 0) / len(df)) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Persentase Linearitas (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "lightgray"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': (linear_stats.get(1, 0) / len(df)) * 100}
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Linear", f"{linear_stats.get(1, 0):,}", 
                         f"{(linear_stats.get(1, 0)/len(df)*100):.1f}%")
                st.metric("Tidak Linear", f"{linear_stats.get(0, 0):,}", 
                         f"{(linear_stats.get(0, 0)/len(df)*100):.1f}%")
            else:
                st.warning("Data linearitas tidak tersedia")
        
        with col2:
            st.markdown('<h3 class="sub-header">Linearitas per Jenjang Pendidikan</h3>', unsafe_allow_html=True)
            
            if 'Pendidikan_Label' in df.columns and 'Linearitas_System_Encoded' in df.columns:
                cross_tab = pd.crosstab(df['Pendidikan_Label'], df['Linearitas_System_Encoded'], 
                                       normalize='index') * 100
                
                fig = px.bar(cross_tab,
                            barmode='group',
                            color_discrete_map={0: '#EF4444', 1: '#10B981'},
                            title="Linearitas per Jenjang Pendidikan (%)",
                            labels={'value': 'Persentase (%)', 
                                   'Pendidikan_Label': 'Jenjang Pendidikan',
                                   'variable': 'Status'})
                
                fig.update_layout(height=400,
                                xaxis_title="Jenjang Pendidikan",
                                yaxis_title="Persentase (%)",
                                xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data pendidikan tidak lengkap")
        
        # ====================================================================
        # FITUR MENAMPILKAN KASUS NYATA DALAM DATAFRAME
        # ====================================================================
        st.markdown('<h3 class="sub-header">Database Kasus Nyata Linearitas Pendidikan-Pekerjaan</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Penjelasan Kolom:</strong><br>
        • Status Linearitas = 0=Tidak Sesuai, 1=Sesuai<br>
        • Jurusan_Cleaned = Hasil standardisasi dari data mentah<br>
        • Bidang_Pekerjaan = Hasil pengelompokkan bidang kerja<br>
        • Status_Pekerjaan = Hasil standardisasi status pekerjaan
        </div>
        """, unsafe_allow_html=True)
        
        # Cek ketersediaan kolom-kolom penting
        required_cols = [
            'Pendidikan terakhir anda',
            'Jurusan_Cleaned_Final_V3',
            'Status_Pekerjaan_Final',
            'Group_Pekerjaan',
            'Linearitas_System_Encoded',
            'Gaji_Kategori',
            'Pendidikan_Label'
        ]
        
        # Filter kolom yang tersedia
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) >= 5:  # Minimal ada 5 kolom penting
            # Tambahkan kolom tambahan jika ada
            additional_cols = []
            for col in ['Jenis Kelamin', 'Kabupaten/Kota domisili anda', 
                       'Status Pernikahan anda', 'Pengalaman_Kerja',
                       'Group_Pekerjaan_Pertama']:
                if col in df.columns:
                    additional_cols.append(col)
            
            all_display_cols = available_cols + additional_cols
            
            # Buat DataFrame untuk display
            display_df = df[all_display_cols].copy()
            
            # Rename kolom untuk tampilan lebih user-friendly
            rename_dict = {
                'Pendidikan terakhir anda': 'Pendidikan_Awal',
                'Jurusan_Cleaned_Final_V3': 'Jurusan',
                'Status_Pekerjaan_Final': 'Status_Pekerjaan',
                'Group_Pekerjaan': 'Bidang_Pekerjaan',
                'Linearitas_System_Encoded': 'Status_Linearitas',
                'Gaji_Kategori': 'Rentang_Gaji',
                'Pendidikan_Label': 'Jenjang_Pendidikan',
                'Jenis Kelamin': 'Gender',
                'Kabupaten/Kota domisili anda': 'Domisili',
                'Status Pernikahan anda': 'Status_Pernikahan',
                'Pengalaman_Kerja': 'Pengalaman',
                'Group_Pekerjaan_Pertama': 'Pekerjaan_Pertama'
            }
            
            display_df = display_df.rename(columns=rename_dict)
            
            # Format Status Linearitas
            if 'Status_Linearitas' in display_df.columns:
                display_df['Status_Linearitas'] = display_df['Status_Linearitas'].map(
                    {0: 'TIDAK SESUAI', 1: 'SESUAI'}
                )
            
            # ====================================================================
            # FILTER INTERAKTIF
            # ====================================================================
            st.markdown("### Filter Data Kasus")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_linearitas = st.selectbox(
                    "Status Linearitas:",
                    options=['Semua', 'Sesuai', 'Tidak Sesuai'],
                    index=0
                )
            
            with col2:
                if 'Jenjang_Pendidikan' in display_df.columns:
                    pendidikan_options = ['Semua'] + sorted(display_df['Jenjang_Pendidikan'].dropna().unique().tolist())
                    filter_pendidikan = st.selectbox(
                        "Jenjang Pendidikan:",
                        options=pendidikan_options,
                        index=0
                    )
                else:
                    filter_pendidikan = 'Semua'
            
            with col3:
                if 'Jurusan' in display_df.columns:
                    jurusan_options = ['Semua'] + sorted(display_df['Jurusan'].dropna().unique().tolist())
                    filter_jurusan = st.selectbox(
                        "Bidang Jurusan:",
                        options=jurusan_options,
                        index=0
                    )
                else:
                    filter_jurusan = 'Semua'

            with col4:
                if 'Bidang_Pekerjaan' in display_df.columns:
                    pekerjaan_options = ['Semua'] + sorted(display_df['Bidang_Pekerjaan'].dropna().unique().tolist())
                    filter_pekerjaan = st.selectbox(
                        "Bidang Pekerjaan:",
                        options=pekerjaan_options,
                        index=0
                    )
                else:
                    filter_pekerjaan = 'Semua'
            
            # Apply filters
            filtered_df = display_df.copy()
            
            if filter_linearitas != 'Semua':
                if filter_linearitas == 'Sesuai':
                    filtered_df = filtered_df[filtered_df['Status_Linearitas'] == 'SESUAI']
                else:
                    filtered_df = filtered_df[filtered_df['Status_Linearitas'] == 'TIDAK SESUAI']
            
            if filter_pendidikan != 'Semua' and 'Jenjang_Pendidikan' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Jenjang_Pendidikan'] == filter_pendidikan]
            
            if filter_jurusan != 'Semua' and 'Jurusan' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Jurusan'] == filter_jurusan]
            
            if filter_pekerjaan != 'Semua' and 'Bidang_Pekerjaan' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Bidang_Pekerjaan'] == filter_pekerjaan]
            
            # ====================================================================
            # TAMPILKAN DATAFRAME DENGAN STYLING
            # ====================================================================
            st.markdown(f"### Menampilkan {len(filtered_df)} Kasus dari {len(display_df)} Total Data")
            
            # Konfigurasi tampilan DataFrame
            pd.set_option('display.max_colwidth', None)
            
            # Buat tabs untuk berbagai view
            view_tabs = st.tabs(["Semua Data", "Analisis Perbandingan", "Statistik Ringkasan"])
            
            with view_tabs[0]:
                # Tampilkan semua data dengan pagination
                rows_per_page = 20
                total_pages = max(1, len(filtered_df) // rows_per_page)
                
                if total_pages > 1:
                    page_number = st.number_input(
                        "Halaman:", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=1,
                        step=1
                    )
                    start_idx = (page_number - 1) * rows_per_page
                    end_idx = min(page_number * rows_per_page, len(filtered_df))
                    
                    st.markdown(f"**Menampilkan data {start_idx + 1}-{end_idx} dari {len(filtered_df)} kasus**")
                    current_data = filtered_df.iloc[start_idx:end_idx]
                else:
                    current_data = filtered_df
                
                # Apply styling untuk highlight mismatch/match
                def highlight_linearitas(row):
                    if row['Status_Linearitas'] == 'TIDAK SESUAI':
                        return ['background-color: #FEE2E2'] * len(row)
                    else:
                        return ['background-color: #DCFCE7'] * len(row)
                
                styled_df = current_data.style.apply(highlight_linearitas, axis=1)
                
                # Tampilkan DataFrame dengan styling
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=600
                )
                
                # Tombol download
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data Filtered (CSV)",
                    data=csv,
                    file_name=f"kasus_linearitas_sumut_{filter_linearitas}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with view_tabs[1]:
                st.markdown("### Analisis Perbandingan Kasus")
                
                # Analisis 1: Distribusi berdasarkan filter
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Status_Linearitas' in filtered_df.columns:
                        distribusi = filtered_df['Status_Linearitas'].value_counts()
                        
                        fig = px.pie(
                            values=distribusi.values,
                            names=distribusi.index,
                            title=f"Distribusi Linearitas (n={len(filtered_df)})",
                            color=distribusi.index,
                            color_discrete_map={
                                'TIDAK SESUAI': '#EF4444',
                                'SESUAI': '#10B981'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Jenjang_Pendidikan' in filtered_df.columns:
                        distribusi_pendidikan = filtered_df['Jenjang_Pendidikan'].value_counts().head(10)
                        
                        fig = px.bar(
                            x=distribusi_pendidikan.values,
                            y=distribusi_pendidikan.index,
                            orientation='h',
                            title="Top 10 Jenjang Pendidikan",
                            color=distribusi_pendidikan.values,
                            color_continuous_scale='blues'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Analisis 2: Cross tab jurusan vs pekerjaan
                st.markdown("### Hubungan Jurusan vs Bidang Pekerjaan")
                
                if 'Jurusan' in filtered_df.columns and 'Bidang_Pekerjaan' in filtered_df.columns:
                    # Ambil top 5 jurusan dan pekerjaan
                    top_jurusan = filtered_df['Jurusan'].value_counts().head(15).index
                    top_pekerjaan = filtered_df['Bidang_Pekerjaan'].value_counts().head(15).index
                    
                    df_cross = filtered_df[
                        filtered_df['Jurusan'].isin(top_jurusan) & 
                        filtered_df['Bidang_Pekerjaan'].isin(top_pekerjaan)
                    ]
                    
                    if len(df_cross) > 0:
                        cross_tab = pd.crosstab(
                            df_cross['Jurusan'], 
                            df_cross['Bidang_Pekerjaan'],
                            normalize='index'
                        ) * 100
                        
                        fig = px.imshow(
                            cross_tab,
                            text_auto='.1f',
                            aspect="auto",
                            color_continuous_scale='RdYlGn',
                            title="Distribusi Bidang Pekerjaan per Jurusan (%)"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            with view_tabs[2]:
                st.markdown("### Statistik Ringkasan Kasus")
                
                # Statistik numerik
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Kasus", f"{len(filtered_df):,}")
                    if 'Status_Linearitas' in filtered_df.columns:
                        match_count = len(filtered_df[filtered_df['Status_Linearitas'] == 'SESUAI'])
                        mismatch_count = len(filtered_df[filtered_df['Status_Linearitas'] == 'TIDAK SESUAI'])
                        st.metric("Kasus Sesuai", f"{match_count:,}", 
                                 f"{(match_count/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "0%")
                        st.metric("Kasus Tidak Sesuai", f"{mismatch_count:,}", 
                                 f"{(mismatch_count/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "0%")
                
                with col2:
                    if 'Jenjang_Pendidikan' in filtered_df.columns:
                        unique_pendidikan = filtered_df['Jenjang_Pendidikan'].nunique()
                        st.metric("Variasi Jenjang", f"{unique_pendidikan}")
                    
                    if 'Jurusan' in filtered_df.columns:
                        unique_jurusan = filtered_df['Jurusan'].nunique()
                        st.metric("Variasi Jurusan", f"{unique_jurusan}")
                    
                    if 'Bidang_Pekerjaan' in filtered_df.columns:
                        unique_pekerjaan = filtered_df['Bidang_Pekerjaan'].nunique()
                        st.metric("Variasi Pekerjaan", f"{unique_pekerjaan}")
                
                with col3:
                    if 'Rentang_Gaji' in filtered_df.columns:
                        # Hitung distribusi gaji
                        gaji_dist = filtered_df['Rentang_Gaji'].value_counts()
                        most_common_gaji = gaji_dist.index[0] if len(gaji_dist) > 0 else "N/A"
                        st.metric("Gaji Paling Umum", most_common_gaji)
                    
                    if 'Pengalaman' in filtered_df.columns:
                        # Hitung distribusi pengalaman
                        exp_dist = filtered_df['Pengalaman'].value_counts()
                        most_common_exp = exp_dist.index[0] if len(exp_dist) > 0 else "N/A"
                        st.metric("Pengalaman Paling Umum", most_common_exp)
                
                # Tampilkan contoh kasus menarik
                st.markdown("### Contoh Kasus Menarik")
                
                if len(filtered_df) > 0:
                    # Contoh kasus mismatch ekstrim
                    if 'TIDAK SESUAI' in filtered_df['Status_Linearitas'].values:
                        mismatch_cases = filtered_df[filtered_df['Status_Linearitas'] == 'TIDAK SESUAI']
                        if len(mismatch_cases) > 0:
                            interesting_mismatch = mismatch_cases.iloc[0]
                            st.markdown(f"""
                            **Contoh Kasus Mismatch:**
                            - **Jurusan:** {interesting_mismatch.get('Jurusan', 'N/A')}
                            - **Bekerja di:** {interesting_mismatch.get('Bidang_Pekerjaan', 'N/A')} ({interesting_mismatch.get('Status_Pekerjaan', 'N/A')})
                            - **Pendidikan:** {interesting_mismatch.get('Jenjang_Pendidikan', 'N/A')}
                            - **Gaji:** {interesting_mismatch.get('Rentang_Gaji', 'N/A')}
                            """)
                    
                    # Contoh kasus match ideal
                    if 'SESUAI' in filtered_df['Status_Linearitas'].values:
                        match_cases = filtered_df[filtered_df['Status_Linearitas'] == 'SESUAI']
                        if len(match_cases) > 0:
                            interesting_match = match_cases.iloc[0]
                            st.markdown(f"""
                            **Contoh Kasus Match:**
                            - **Jurusan:** {interesting_match.get('Jurusan', 'N/A')}
                            - **Bekerja di:** {interesting_match.get('Bidang_Pekerjaan', 'N/A')} ({interesting_match.get('Status_Pekerjaan', 'N/A')})
                            - **Pendidikan:** {interesting_match.get('Jenjang_Pendidikan', 'N/A')}
                            - **Gaji:** {interesting_match.get('Rentang_Gaji', 'N/A')}
                            """)
            
            # ====================================================================
            # ANALISIS TREN BERDASARKAN DATA NYATA
            # ====================================================================
            st.markdown("### Analisis Tren Berdasarkan Data Nyata")
            
            # Tren 1: Persentase linearitas per jurusan
            if 'Jurusan' in filtered_df.columns and 'Status_Linearitas' in filtered_df.columns:
                # Ambil jurusan dengan minimal 10 kasus
                jurusan_counts = filtered_df['Jurusan'].value_counts()
                significant_jurusan = jurusan_counts[jurusan_counts >= 10].index
                
                if len(significant_jurusan) > 0:
                    jurusan_linearity = []
                    for jurusan in significant_jurusan:
                        jurusan_data = filtered_df[filtered_df['Jurusan'] == jurusan]
                        total = len(jurusan_data)
                        match = len(jurusan_data[jurusan_data['Status_Linearitas'] == 'SESUAI'])
                        match_percentage = (match / total * 100) if total > 0 else 0
                        
                        jurusan_linearity.append({
                            'Jurusan': jurusan,
                            'Total_Kasus': total,
                            '%_Sesuai': match_percentage,
                            '%_Tidak_Sesuai': 100 - match_percentage
                        })
                    
                    jurusan_df = pd.DataFrame(jurusan_linearity).sort_values('%_Sesuai', ascending=False)
                    
                    fig = px.bar(
                        jurusan_df.head(10),
                        x='Jurusan',
                        y=['%_Sesuai', '%_Tidak_Sesuai'],
                        title="Top 10 Jurusan dengan Tingkat Kesesuaian Tertinggi",
                        barmode='stack',
                        color_discrete_map={
                            '%_Sesuai': '#10B981',
                            '%_Tidak_Sesuai': '#EF4444'
                        }
                    )
                    fig.update_layout(xaxis_tickangle=45, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tren 2: Hubungan pendidikan dengan linearitas
            if 'Jenjang_Pendidikan' in filtered_df.columns and 'Status_Linearitas' in filtered_df.columns:
                edu_linearity = filtered_df.groupby('Jenjang_Pendidikan').agg(
                    Total=('Status_Linearitas', 'count'),
                    Sesuai=('Status_Linearitas', lambda x: (x == 'SESUAI').sum())
                )
                edu_linearity['%_Sesuai'] = (edu_linearity['Sesuai'] / edu_linearity['Total'] * 100).round(1)
                
                fig = px.scatter(
                    edu_linearity,
                    x='Total',
                    y='%_Sesuai',
                    size='Total',
                    color='%_Sesuai',
                    hover_name=edu_linearity.index,
                    title="Hubungan Jumlah Kasus vs Tingkat Kesesuaian per Jenjang Pendidikan",
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("""
            Data tidak lengkap untuk menampilkan kasus nyata
            
            Kolom yang diperlukan:
            1. Pendidikan terakhir anda
            2. Jurusan_Cleaned_Final_V3
            3. Status_Pekerjaan_Final
            4. Group_Pekerjaan
            5. Linearitas_System_Encoded
            
            Pastikan proses data cleaning sudah berjalan dengan benar.
            """)
            
# 6. PREDIKSI MODEL ML
elif selected_page == "Prediksi Model ML":
    st.markdown('<h2 class="sub-header">Prediksi Linearitas dengan Machine Learning</h2>', unsafe_allow_html=True)
    
    if rf_model is not None and nb_model is not None and df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">Performa Model</h3>', unsafe_allow_html=True)
            
            # Metrics untuk kedua model
            metrics_data = {
                'Model': ['Random Forest', 'Naive Bayes'],
                'Akurasi': [0.70, 0.6],
                'Presisi': [0.72, 0.68],
                'Recall': [0.80, 0.75],
                'F1-Score': [0.76, 0.71]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = go.Figure(data=[
                go.Bar(name='Random Forest',
                      x=metrics_df.columns[1:],
                      y=metrics_df.iloc[0, 1:],
                      marker_color='#3B82F6'),
                go.Bar(name='Naive Bayes',
                      x=metrics_df.columns[1:],
                      y=metrics_df.iloc[1, 1:],
                      marker_color='#10B981')
            ])
            
            fig.update_layout(barmode='group',
                            height=400,
                            title="Perbandingan Performa Model ML")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Random Forest lebih unggul dengan akurasi")
        
        with col2:
            st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
            
            # Data importance dari analisis
            importance_data = {
                'Faktor': ['Jenjang Pendidikan', 'Jurusan', 'Domisili', 'Gender', 'Pernikahan'],
                'Importance': [0.936, 0.042, 0.012, 0.006, 0.004]
            }
            
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df,
                        x='Importance',
                        y='Faktor',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='viridis',
                        text='Importance')
            
            fig.update_layout(height=400,
                            title="Faktor Penentu Linearitas",
                            xaxis_title="Tingkat Pengaruh",
                            yaxis_title="Faktor")
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Insight: Pendidikan adalah faktor terpenting")

# 7. INSIGHTS & REKOMENDASI
elif selected_page == "Insights & Rekomendasi":
    st.markdown('<h1 class="main-header">Insights & Rekomendasi Strategis</h1>', unsafe_allow_html=True)
    
    if df is not None:
        total_responden = len(df)
        total_non_linear = len(df[df['Linearitas_System_Encoded'] == 0]) if 'Linearitas_System_Encoded' in df.columns else 0
        
        # Executive Summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center;">Executive Summary</h2>
        <p style="font-size: 1.2rem; text-align: center;">
        Analisis terhadap <strong>{total_responden:,} responden</strong> di Sumatera Utara menunjukkan bahwa 
        <strong>{(total_non_linear/total_responden*100):.1f}% bekerja di luar bidang pendidikannya</strong>. 
        </p>
        </div>
        """, unsafe_allow_html=True)

    # Call to Action
    if df is not None and 'Linearitas_System_Encoded' in df.columns:
        total_non_linear = len(df[df['Linearitas_System_Encoded'] == 0])
        percentage = (total_non_linear / len(df)) * 100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-top: 2rem; text-align: center;">
        <h2 style="color: white;">Siap Mengurangi Mismatch Pendidikan-Pekerjaan?</h2>
        <p style="font-size: 1.1rem;">
        <strong>{percentage:.1f}% mismatch</strong> adalah peluang untuk transformasi besar. 
        Mari wujudkan Sumatera Utara dengan SDM yang tepat guna dan produktif!
        </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Analisis Linearitas Pendidikan vs Pekerjaan di Sumatera Utara | © Kelompok 04</p>
</div>
""", unsafe_allow_html=True)