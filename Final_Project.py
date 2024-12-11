import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as s
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

st.title("Final Project SDBDSS")
st.write("Rihhadatul Aisy Nadhilah")
st.write("5023211020")

# Sidebar menu untuk navigasi
menu = st.sidebar.radio("Menu", ["Data Preparation", "Feature Selection", "Down Sampling", "Split Data", "Classification"])
# Load data
data_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if menu == "Data Preparation":
    st.write("Data Preparation")

    if data_file is not None:
        data = pd.read_excel(data_file)
                
        # Tampilkan data
        st.write("### Data yang diunggah:")
        st.dataframe(data)
        
        # Tampilkan kolom data
        st.write("### Kolom Data:")
        st.write(data.columns.tolist())
        
        # Informasi statistik tentang data
        st.write("### Informasi Data:")
        buffer = []
        data.info(buf=buffer)
        st.text("\n".join(buffer))

        # Cek nilai yang hilang
        st.write("### Cek Missing Values:")
        missing_values = data.isnull().sum().sort_values(ascending=False)
        st.write(missing_values)

        # Distribusi kelas
        st.write("### Distribusi Kelas:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='class', data=data, palette='viridis', ax=ax)
        ax.set_title('Distribusi Data Kelas')
        ax.set_xlabel('Kelas')
        ax.set_ylabel('Jumlah Data')
        st.pyplot(fig)

        # Filter data hanya untuk kelas 0 dan 3
        st.write("### Filter Data untuk Kelas 0 dan 3:")
        filtered_data = data[data['class'].isin([0, 3])]
        st.dataframe(filtered_data)

elif menu == "Feature Selection":
    st.write("Anda berada di menu Feature Selection")
    if 'filtered_data' in locals():
        # Drop kolom file_name
        filtered_data = filtered_data.drop(columns=['file_name'])

        # Plot menggunakan Seaborn
        st.write("### Visualisasi Distribusi Fitur")
        long_data = pd.melt(
            filtered_data, 
            id_vars="class",  # Kolom untuk pengelompokan
            value_vars=[col for col in filtered_data.columns if col != "class"],  # Semua kolom kecuali `class`
            var_name="feature",
            value_name="value"
        )

        g = sns.FacetGrid(long_data, col="feature", hue="class", 
                          sharex=False, sharey=False, col_wrap=5)
        g.map(sns.kdeplot, "value", shade=True)
        st.pyplot(g.fig)

        # Hitung korelasi data
        st.write("### Korelasi Data")
        data_copy = filtered_data.replace(to_replace=['0', '3'], value=[0, 1], inplace=False)
        corr_data = data_copy.corr()
        st.write(corr_data)

        # Heatmap
        st.write("### Heatmap Korelasi")
        mask = np.zeros_like(corr_data)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(35, 15))
            sns.heatmap(data=corr_data, vmin=0, vmax=1, mask=mask, square=True, annot=True, ax=ax)
            st.pyplot(f)

        # Fitur dengan korelasi signifikan
        st.write("### Fitur Signifikan")
        strong_relation_features = pd.Series(corr_data['class']).nlargest(n=7).iloc[1:]
        st.write(strong_relation_features)

        # Data dengan fitur signifikan
        diagnosis = data_copy['class']
        data_copy = data_copy[list(strong_relation_features.to_dict().keys())]
        data_copy['class'] = diagnosis

        st.write("### Data dengan Fitur Signifikan:")
        st.dataframe(data_copy)

        # Heatmap ulang untuk fitur signifikan
        st.write("### Heatmap Korelasi Fitur Signifikan")
        mask = np.zeros_like(data_copy.corr())
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(data=data_copy.corr(), vmin=0, vmax=1, mask=mask, square=True, annot=True, ax=ax)
            st.pyplot(f)
    else:
        st.write("Data belum tersedia. Silakan upload data di menu Data Preparation.")


elif menu == "Down Sampling":
    st.write("Anda berada di menu Down Sampling")
    if 'data_copy' in locals():
        # Pisahkan data berdasarkan kelas
        class_0 = data_copy[data_copy['class'] == 0]
        class_3 = data_copy[data_copy['class'] == 1]

        # Hitung jumlah minoritas
        minor = len(class_3)

        # Downsample kelas mayoritas (class_0)
        class_0_downsampled = resample(
            class_0, 
            replace=False,      # Sample tanpa pengembalian
            n_samples=minor,    # Sesuaikan ukuran dengan jumlah minoritas
            random_state=42     # Hasil reproducible
        )

        # Gabungkan kembali data
        data_fix = pd.concat([class_0_downsampled, class_3])

        # Acak urutan data
        data_fix = data_fix.sample(frac=1, random_state=42).reset_index(drop=True)

        # Tampilkan hasil
        st.write("### Data setelah Down Sampling:")
        st.dataframe(data_fix)

        st.write("Jumlah data untuk setiap kelas:")
        st.write("Kelas 0:", data_fix[data_fix['class'] == 0].shape[0])
        st.write("Kelas 1:", data_fix[data_fix['class'] == 1].shape[0])

        # Pisahkan fitur dan target
        x = data_fix.drop('class', axis=1)  # Fitur
        y = data_fix['class']               # Kelas target
    else:
        st.write("Data belum tersedia. Silakan lakukan Feature Selection terlebih dahulu.")

elif menu == "Split Data":
    st.write("Anda berada di menu Split Data")
    if 'data_fix' in locals():
        # Pisahkan data berdasarkan kelas
        class0_data = data_fix[data_fix['class'] == 0]
        class3_data = data_fix[data_fix['class'] == 1]

        # Split data menjadi training dan testing
        class0_training_data = class0_data.iloc[0:int(0.75*len(class0_data))]
        class3_training_data = class3_data.iloc[0:int(0.75*len(class3_data))]

        class0_testing_data = class0_data.iloc[int(0.75*len(class0_data)):]
        class3_testing_data = class3_data.iloc[int(0.75*len(class3_data)):]

        training_data = pd.concat([class0_training_data, class3_training_data])
        testing_data = pd.concat([class0_testing_data, class3_testing_data])

        # Ubah nilai 3 menjadi 1 di kolom 'class'
        training_data.loc[training_data['class'] == 3, 'class'] = 1
        testing_data.loc[testing_data['class'] == 3, 'class'] = 1

        # Tampilkan hasil
        st.write("### Data Training:")
        st.dataframe(training_data)

        st.write("### Data Testing:")
        st.dataframe(testing_data)

        st.write("Jumlah data pada masing-masing subset:")
        st.write("Training data kelas 0:", len(class0_training_data))
        st.write("Training data kelas 1:", len(class3_training_data))
        st.write("Testing data kelas 0:", len(class0_testing_data))
        st.write("Testing data kelas 1:", len(class3_testing_data))
    else:
        st.write("Data belum tersedia. Silakan lakukan Down Sampling terlebih dahulu.")


elif menu == "Classification":
    st.write("Anda berada di menu Classification")

    # Sub-menu untuk Classification
    sub_menu = st.radio("Pilih Sub-Menu Classification", ["BAYESIAN", "NAIVE BAYES", "LOGISTIC REGRESSION", "DECISION TREE", "Comparation"])

    if sub_menu == "BAYESIAN":
        st.write("Anda memilih Model 1")
        # Tambahkan kode untuk Model 1

    elif sub_menu == "NAIVE BAYES":
        st.write("Anda memilih Model 2")
        # Tambahkan kode untuk Model 2

    elif sub_menu == "LOGISTIC REGRESSION":
        st.write("Anda memilih Model 3")
        # Tambahkan kode untuk Model 3

    elif sub_menu == "DECISION TREE":
        st.write("Anda memilih Model 4")
        # Tambahkan kode untuk Model 4

    elif sub_menu == "Comparation":
        st.write("Anda memilih Model 4")
        # Tambahkan kode untuk Model 4