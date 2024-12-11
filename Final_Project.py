import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as s
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from celluloid import Camera
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import logit model
import os
import sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)
from LR import LogisticRegression
from DT import DecisionTree
from collections import defaultdict
from scipy.stats import norm


st.title("Final Project SDBDSS")
st.write("Rihhadatul Aisy Nadhilah")
st.write("5023211020")

# Fungsi untuk plot distribusi data dengan cache
@st.cache_data
def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='class', data=data, palette='viridis', ax=ax)
    ax.set_title('Distribusi Data Kelas')
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Jumlah Data')
    return fig

def predict_classes_naive(data):
    
    p_xi_on_class1 = s.multivariate_normal.pdf(data,mu_1,sigma_1)    
    p_xi_on_class0 = s.multivariate_normal.pdf(data,mu_0,sigma_0)
    p_class1_on_xi = p_xi_on_class1/(p_xi_on_class0 + p_xi_on_class1)
    
    return p_class1_on_xi > 0.5

import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.header = [
            "objects HM", "min axis EX", "contrast HM", 
            "perimeter EX", "maj axis EX"
        ]
    
    def is_numeric(self, value):
        return isinstance(value, (int, float))
    
    def unique_vals(self, rows, col):
        return set([row[col] for row in rows])
    
    def class_counts(self, rows):
        counts = {}
        for row in rows:
            label = row[-1]
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    class Question:
        def __init__(self, column, value):
            self.column = column
            self.value = value
        
        def match(self, example):
            val = example[self.column]
            if isinstance(val, (int, float)):
                return val >= self.value
            else:
                return val == self.value
        
        def __repr__(self, header):
            condition = "==" if not isinstance(self.value, (int, float)) else ">="
            return f"Is {header[self.column]} {condition} {str(self.value)}?"
    
    def partition(self, rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
    
    def gini(self, rows):
        counts = self.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity
    
    def info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)
    
    def find_best_split(self, rows):
        best_gain = 0
        best_question = None
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1
        
        for col in range(n_features):
            values = set([row[col] for row in rows])
            for val in values:
                question = self.Question(col, val)
                true_rows, false_rows = self.partition(rows, question)
                
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                
                gain = self.info_gain(true_rows, false_rows, current_uncertainty)
                
                if gain >= best_gain:
                    best_gain, best_question = gain, question
        
        return best_gain, best_question
    
    class Leaf:
        def __init__(self, rows):
            self.predictions = {}
            counts = {}
            for row in rows:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1
            
            total = sum(counts.values())
            for label, count in counts.items():
                self.predictions[label] = count / total
    
    class DecisionNode:
        def __init__(self, question, true_branch, false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch
    
    def build_tree(self, rows, depth=0):
        gain, question = self.find_best_split(rows)
        
        if self.max_depth is not None and depth >= self.max_depth:
            return self.Leaf(rows)
        
        if gain == 0:
            return self.Leaf(rows)
        
        true_rows, false_rows = self.partition(rows, question)
        
        true_branch = self.build_tree(true_rows, depth + 1)
        false_branch = self.build_tree(false_rows, depth + 1)
        
        return self.DecisionNode(question, true_branch, false_branch)
    
    def fit(self, X, y):
        # Combine features and labels
        training_data = [list(x) + [label] for x, label in zip(X, y)]
        self.tree = self.build_tree(training_data)
        return self
    
    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self.classify(row, self.tree))
        return predictions
    
    def classify(self, row, node):
        if isinstance(node, self.Leaf):
            return max(node.predictions, key=node.predictions.get)
        
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)
    
    def accuracy_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)


# Sidebar menu untuk navigasi
menu = st.sidebar.radio("Menu", ["Data Preparation", "Feature Selection", "Down Sampling", "Split Data", "Classification"])

# Inisialisasi session state untuk menyimpan data
if "data" not in st.session_state:
    st.session_state.data = None
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None
if "data_copy" not in st.session_state:
    st.session_state.data_copy = None
if "data_fix" not in st.session_state:
    st.session_state.data_fix = None
if "training_data" not in st.session_state:
    st.session_state.training_data = None
if "testing_data" not in st.session_state:
    st.session_state.testing_data = None


if menu == "Data Preparation":
    st.write("### Data Preparation")
    # Load data
    data_file = st.file_uploader("Upload file Excel", type=["xlsx"])

    if data_file is not None:
        data = pd.read_excel(data_file)
        st.session_state.data = data
                
        # Tampilkan data
        st.write("### Data yang diunggah:")
        st.dataframe(data)
        
        # Tampilkan kolom data
        st.write("### Kolom Data:")
        st.write(data.columns.tolist())

        # Distribusi kelas
        st.write("### Distribusi Kelas:")
        st.pyplot(plot_distribution(data))

        # Filter data hanya untuk kelas 0 dan 3
        st.write("### Filter Data untuk Kelas 0 dan 3:")
        filtered_data = data[data['class'].isin([0, 3])]
        st.session_state.filtered_data = filtered_data
        st.dataframe(filtered_data)

elif menu == "Feature Selection":
    st.write("### Feature Selection")
    if st.session_state.data is not None:
        filtered_data = st.session_state.filtered_data

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
        data_copy = filtered_data.copy()
        data_copy['class'] = data_copy['class'].replace({0: 0, 3: 1})
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
        st.session_state.data_copy = data_copy

        # Heatmap ulang untuk fitur signifikan
        st.write("### Heatmap Korelasi Fitur Signifikan")
        mask = np.zeros_like(data_copy.corr())
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(data=data_copy.corr(), vmin=0, vmax=1, mask=mask, square=True, annot=True, ax=ax)
            st.pyplot(f)

        st.write("### Data dengan Fitur Signifikan:")
        st.dataframe(data_copy)
    else:
        st.write("Data belum tersedia. Silakan upload data di menu Data Preparation.")


elif menu == "Down Sampling":
    st.write("### Down Sampling")
    if st.session_state.data_copy is not None:
        data_copy = st.session_state.data_copy

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
        st.session_state.data_fix = data_fix
        
        # Tampilkan hasil
        st.write("### Data setelah Down Sampling:")
        st.dataframe(data_fix)

        st.write("Jumlah data untuk setiap kelas:")
        st.write("Kelas 0:", data_fix[data_fix['class'] == 0].shape[0])
        st.write("Kelas 1:", data_fix[data_fix['class'] == 1].shape[0])

        # Pisahkan fitur dan target
        #x = data_fix.drop('class', axis=1)  # Fitur
        #y = data_fix['class']               # Kelas target
    else:
        st.write("Data belum tersedia. Silakan lakukan Feature Selection terlebih dahulu.")

elif menu == "Split Data":
    st.write("### Split Data")
    if st.session_state.data_fix is not None:
        data_fix = st.session_state.data_fix

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
    st.write("## **Classification**")

    # Sub-menu untuk Classification
    sub_menu = st.radio("Pilih Sub-Menu Classification", ["BAYESIAN", "NAIVE BAYES", "LOGISTIC REGRESSION", "DECISION TREE"])

    if sub_menu == "BAYESIAN":
        st.write("### **Bayesian**")
        if st.session_state.training_data is not None and st.session_state.testing_data is not None:
            training_data = st.session_state.training_data
            testing_data = st.session_state.testing_data
            x_train = training_data.drop('class', axis=1).to_numpy()  # Fitur
            y_train = training_data['class'].to_numpy()              # Kelas
            x_test = testing_data.drop('class', axis=1).to_numpy()  # Fitur
            y_test = testing_data['class'].to_numpy() # Target Sebenarnya

            classes = np.unique(y_train)  # [0, 1]

            # Menyimpan mean dan std untuk setiap kelas
            class_stats = {}
            for cls in classes:
                # Filter data berdasarkan kelas
                data_cls = x_train[y_train == cls]
                # Hitung mean dan std untuk setiap fitur
                mean = np.mean(data_cls, axis=0)
                std = np.std(data_cls, axis=0)
                # Simpan statistik
                class_stats[cls] = {'mean': mean, 'std': std}

            # Fungsi untuk menghitung PDF
            def compute_pdf(x, mean, std):
                return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2) / (2 * std**2))
            
            def normalize_pdf(pdf):
                pdf_min = pdf.min()
                pdf_max = pdf.max()
                return (pdf - pdf_min) / (pdf_max - pdf_min)

            features = training_data.drop(columns=["class"])  # Pisahkan fitur dari label
            labels = training_data["class"]  # Kolom label

            # Ambil nama fitur langsung dari training_data
            feature_names = features.columns

            # Mengambil kelas dari class_stats
            classes = class_stats.keys()

            # Menentukan rentang nilai x berdasarkan data x_train
            x_min = np.min(x_train, axis=0)
            x_max = np.max(x_train, axis=0)
            x_values = np.linspace(x_min, x_max, 100)  # Menentukan rentang nilai berdasarkan data

            for i, feature_name in enumerate(feature_names):
                # Ambil mean dan std untuk Class 0 dan Class 1
                mean_0 = class_stats[0]['mean'][i]
                std_0 = class_stats[0]['std'][i]
                mean_1 = class_stats[1]['mean'][i]
                std_1 = class_stats[1]['std'][i]

                # Hitung PDF untuk Class 0 dan Class 1 pada fitur i
                pdf_0 = compute_pdf(x_values, mean_0, std_0)
                pdf_1 = compute_pdf(x_values, mean_1, std_1)

            def compute_error_probability(mean_0, std_0, mean_1, std_1, threshold):
                """
                Menghitung error probability dengan menggunakan rumus yang diberikan.
                """
                alpha = norm.cdf((threshold - mean_0) / std_0)
                beta = 1 - norm.cdf((threshold - mean_1) / std_1)
                return alpha, beta

            def compute_double_threshold(alpha, beta):
                """
                Menghitung double threshold menggunakan rumus yang diberikan.
                """
                gamma_1 = (1 - beta) / alpha
                gamma_2 = beta / (1 - alpha)
                return gamma_1, gamma_2
            
            mean_0 = class_stats[0]['mean'][i]
            std_0 = class_stats[0]['std'][i]
            mean_1 = class_stats[1]['mean'][i]
            std_1 = class_stats[1]['std'][i]
            threshold = (mean_0 + mean_1) / 2  # Contoh: threshold sebagai rata-rata dua distribusi

            # Menghitung error probability
            alpha, beta = compute_error_probability(mean_0, std_0, mean_1, std_1, threshold)
            # Menghitung double threshold
            gamma_1, gamma_2 = compute_double_threshold(alpha, beta)
            
            def calculate_probability_ratio(x, mean1, std1, mean2, std2):
                pdf1 = compute_pdf(x, mean1, std1)
                pdf2 = compute_pdf(x, mean2, std2)
                return pdf1 / pdf2
            
            probability_ratio = calculate_probability_ratio(x_values, mean_0, std_0, mean_1, std_1)

            # Fungsi untuk melakukan prediksi
            def predict(x_test, class_stats, priors=None):
                predictions = []
                for x in x_test:  # Untuk setiap sampel dalam testing data
                    posteriors = []
                    for cls, stats in class_stats.items():
                        # Hitung likelihood: P(X|C)
                        likelihood = np.prod(compute_pdf(x, stats['mean'], stats['std']))
                        
                        # Hitung prior: P(C) (default: sama untuk semua kelas jika tidak ditentukan)
                        prior = priors[cls] if priors else 1 / len(class_stats)
                        
                        # Hitung posterior: P(C|X) = P(X|C) * P(C)
                        posterior = likelihood * prior
                        posteriors.append((cls, posterior))
                    
                    # Pilih kelas dengan posterior tertinggi
                    predicted_class = max(posteriors, key=lambda x: x[1])[0]
                    predictions.append(predicted_class)
                return np.array(predictions)
            
            # Hitung prior probability (opsional)
            priors = {cls: np.sum(y_train == cls) / len(y_train) for cls in classes}

            # Prediksi untuk testing data
            y_pred_bayes = predict(x_test, class_stats, priors)

            # Hitung confusion matrix
            cm_bayes = confusion_matrix(y_test, y_pred_bayes)

            # Buat DataFrame untuk representasi tabel
            cm_df_bayes = pd.DataFrame(cm_bayes, index=classes, columns=classes)

            # Tampilkan confusion matrix
            st.write("### Confusion Matrix:")
            st.dataframe(cm_df_bayes)

            # Evaluasi akurasi
            accuracy_bayes = np.sum(y_pred_bayes == y_test) / len(y_test)
            st.write(f"### Akurasi: {accuracy_bayes * 100:.2f}%")

            # Classification report
            st.write("### Classification Report:")
            st.text(classification_report(y_test, y_pred_bayes))
        else:
            st.write("Data untuk training dan testing belum tersedia. Silakan lakukan Split Data terlebih dahulu.")

    elif sub_menu == "NAIVE BAYES":
        st.write("### **Naive Bayes**")
        if st.session_state.training_data is not None and st.session_state.testing_data is not None:
            training_data = st.session_state.training_data
            testing_data = st.session_state.testing_data

            # Konversi DataFrame ke NumPy array
            training_data_np = training_data.to_numpy()
            testing_data_np = testing_data.to_numpy()

            # Indeks fitur
            features = list(range(training_data_np.shape[1] - 1))  # Semua kecuali kolom terakhir
            class_column = -1  # Kolom terakhir sebagai class label

            # Data untuk kelas 0
            class_0_data = training_data_np[training_data_np[:, class_column] == 0][:, features]
            mu_0 = class_0_data.mean(axis=0)
            sigma_0 = np.cov(class_0_data, rowvar=False)

            # Data untuk kelas 1
            class_1_data = training_data_np[training_data_np[:, class_column] == 1][:, features]
            mu_1 = class_1_data.mean(axis=0)
            sigma_1 = np.cov(class_1_data, rowvar=False)

            def predict_classes(data):
                p_xi_on_class1 = s.multivariate_normal.pdf(data, mu_1, sigma_1)
                p_xi_on_class0 = s.multivariate_normal.pdf(data, mu_0, sigma_0)
                p_class1_on_xi = p_xi_on_class1 / (p_xi_on_class0 + p_xi_on_class1)
                return p_class1_on_xi > 0.5

            features_data = testing_data_np[:, 0:6]  # 6 fitur
            predicted_classes_nb = predict_classes(features_data)

            # Evaluasi
            y_true_nb = testing_data.iloc[:, -1].to_numpy()
            accuracy_nb = accuracy_score(y_true_nb, predicted_classes_nb) * 100
            conf_matrix_nb = confusion_matrix(y_true_nb, predicted_classes_nb)

            # Tampilkan hasil evaluasi
            st.write("### Confusion Matrix:")
            st.dataframe(pd.DataFrame(conf_matrix_nb, index=[0, 1], columns=[0, 1]))
            st.write(f"### Akurasi: {accuracy_nb:.2f}%")
            st.write("### Classification Report:")
            st.text(classification_report(y_true_nb, predicted_classes_nb))
        else:
            st.write("Data untuk training dan testing belum tersedia. Silakan lakukan Split Data terlebih dahulu.")

    elif sub_menu == "LOGISTIC REGRESSION":
        st.write("### **Logistic Regression**")
        if st.session_state.training_data is not None and st.session_state.testing_data is not None:
            training_data = st.session_state.training_data
            testing_data = st.session_state.testing_data

            # Memisahkan fitur (X) dan label (y) untuk data latih
            x_train = training_data[:, :-1]  # Semua kolom kecuali kolom terakhir
            y_train = training_data[:, -1]   # Kolom terakhir sebagai label

            # Memisahkan fitur (X) dan label (y) untuk data uji
            x_test = testing_data[:, :-1]    # Semua kolom kecuali kolom terakhir
            y_test = testing_data[:, -1]     # Kolom terakhir sebagai label

            # Kolom yang ingin dinormalisasi adalah kolom pertama hingga ke-5 (kolom terakhir adalah 'class')
            features_to_standardize = ['objects HM', 'min axis EX', 'contrast HM', 'perimeter EX', 'maj axis EX']

            column_transformer = ColumnTransformer(
                [("scaler", StandardScaler(), features_to_standardize)], remainder="passthrough"
            )

            # Normalisasi data latih dan data uji
            x_train = column_transformer.fit_transform(training_data)
            x_test = column_transformer.transform(testing_data)

            # Inisialisasi model Logistic Regression
            model = LogisticRegression(n_input_features=x_train.shape[-1])

            # Latih model dengan data latih
            costs, accuracies, weights, bias = model.train(
                x_train, y_train,
                epochs=5000,
                learning_rate=0.01,
                minibatch_size=None,
                verbose=True
            )

            # Prediksi label pada data uji
            predictions_lr = model.predict(x_test)
            predictions_lr = (predictions_lr > 0.5).astype(int)

            # Evaluasi
            accuracy_lr = model.accuracy(predictions_lr, y_test.astype(int))
            cm_lr = confusion_matrix(y_test.astype(int), predictions_lr)
            cm_df = pd.DataFrame(cm_lr, index=np.unique(y_test), columns=np.unique(y_test))

            # Tampilkan hasil evaluasi
            st.write("### Confusion Matrix:")
            st.dataframe(cm_df)
            st.write(f"### Akurasi: {accuracy_lr:.2f}%")
            st.write("### Classification Report:")
            st.text(classification_report(y_test.astype(int), predictions_lr))
        else:
            st.write("Data untuk training dan testing belum tersedia. Silakan lakukan Split Data terlebih dahulu.")
            
    elif sub_menu == "DECISION TREE":
        st.write("### **Decision Tree**")
        if st.session_state.training_data is not None and st.session_state.testing_data is not None:
            training_data = st.session_state.training_data
            testing_data = st.session_state.testing_data

            training_data_np = training_data.to_numpy()  # Konversi ke NumPy array
            testing_data_np = testing_data.to_numpy()    # Konversi ke NumPy array

            # Memisahkan fitur (X) dan label (y) untuk data latih
            x_train = training_data_np[:, :-1]  # Semua kolom kecuali kolom terakhir
            y_train = training_data_np[:, -1]   # Kolom terakhir sebagai label

            # Memisahkan fitur (X) dan label (y) untuk data uji
            x_test = testing_data_np[:, :-1]    # Semua kolom kecuali kolom terakhir
            y_test = testing_data_np[:, -1]     # Kolom terakhir sebagai label

            # Inisialisasi model Decision Tree
            dt = DecisionTreeClassifier(max_depth=5)

            # Melatih model
            dt.fit(x_train, y_train)

            # Prediksi label
            predictions_dt = dt.predict(x_test)

            # Evaluate the model
            cm_dt = confusion_matrix(y_test, predictions_dt)
            accuracy_dt = accuracy(y_test, predictions_dt)
            
            st.write("### Confusion Matrix:")
            st.dataframe(cm_dt)
            st.write(f"### Akurasi: {accuracy_dt:.2f}%")
            st.write("### Classification Report:")
            st.text(classification_report(y_test.astype(int), predictions_dt))
        else:
            st.write("Data untuk training dan testing belum tersedia. Silakan lakukan Split Data terlebih dahulu.")
  
