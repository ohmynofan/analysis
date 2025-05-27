import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

try:
    df = pd.read_csv('data_siswa.csv')
    print("Data berhasil dimuat. Jumlah baris:", len(df))
except FileNotFoundError:
    print("File tidak ditemukan. Pastikan file 'data_siswa.csv' ada di direktori yang sama.")
    exit()
except Exception as e:
    print("Terjadi kesalahan saat membaca file:", e)
    exit()

df.columns = df.columns.str.strip()

relevant_cols = ['Durasi', 'JumlahStress', 'JumlahCemas']
df = df[relevant_cols].copy()

df = df.dropna()

durasi_mapping = {
    'Kurang dari 2 jam': 1,
    '2-4 jam': 3,
    'Lebih dari 4 jam': 5
}
df['Durasi_numerik'] = df['Durasi'].map(durasi_mapping)

df['JumlahStress'] = pd.to_numeric(df['JumlahStress'], errors='coerce')
df['JumlahCemas'] = pd.to_numeric(df['JumlahCemas'], errors='coerce')
df = df.dropna()  

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Durasi_numerik', 'JumlahStress', 'JumlahCemas']])

wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), wcss, marker='o', linestyle='--')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS')
plt.show()

k = 2
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df['cluster'] = clusters

silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"Silhouette Score: {silhouette_avg:.2f}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Durasi_numerik', y='JumlahStress', hue='cluster', palette='viridis', s=100)
plt.xticks([1, 3, 5], ['<2 jam', '2-4 jam', '>4 jam'])
plt.title('Tingkat Stres vs Durasi Penggunaan Gadget')
plt.xlabel('Durasi Penggunaan Gadget')
plt.ylabel('Tingkat Stres (1-5)')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Durasi_numerik', y='JumlahCemas', hue='cluster', palette='viridis', s=100)
plt.xticks([1, 3, 5], ['<2 jam', '2-4 jam', '>4 jam'])
plt.title('Tingkat Cemas vs Durasi Penggunaan Gadget')
plt.xlabel('Durasi Penggunaan Gadget')
plt.ylabel('Tingkat Cemas (1-5)')
plt.legend()

plt.tight_layout()
plt.show()

cluster_analysis = df.groupby('cluster').agg({
    'Durasi_numerik': ['mean', 'min', 'max'],
    'JumlahStress': ['mean', 'min', 'max'],
    'JumlahCemas': ['mean', 'min', 'max'],
    'cluster': 'count'
})

print("\nAnalisis Cluster:")
print(cluster_analysis)

print("\nPerbandingan Durasi Penggunaan Gadget:")
df['durasi_kategori'] = df['Durasi'].apply(lambda x: '2-4 jam' if x == '2-4 jam' else ('>4 jam' if x == 'Lebih dari 4 jam' else '<2 jam'))
comparison = df.groupby('durasi_kategori').agg({
    'JumlahStress': 'mean',
    'JumlahCemas': 'mean'
}).sort_values('JumlahStress', ascending=False)

print(comparison)

print("\nRata-rata Tingkat Stres dan Cemas berdasarkan Durasi:")
stats = df.groupby('durasi_kategori').agg({
    'JumlahStress': ['mean', 'median', 'std'],
    'JumlahCemas': ['mean', 'median', 'std']
})
print(stats)

from scipy.stats import f_oneway

print("\nUji Signifikansi Perbedaan Tingkat Stres:")
f_val, p_val = f_oneway(
    df[df['durasi_kategori'] == '<2 jam']['JumlahStress'],
    df[df['durasi_kategori'] == '2-4 jam']['JumlahStress'],
    df[df['durasi_kategori'] == '>4 jam']['JumlahStress']
)
print(f"F-value: {f_val:.2f}, p-value: {p_val:.4f}")

print("\nUji Signifikansi Perbedaan Tingkat Cemas:")
f_val, p_val = f_oneway(
    df[df['durasi_kategori'] == '<2 jam']['JumlahCemas'],
    df[df['durasi_kategori'] == '2-4 jam']['JumlahCemas'],
    df[df['durasi_kategori'] == '>4 jam']['JumlahCemas']
)
print(f"F-value: {f_val:.2f}, p-value: {p_val:.4f}")