import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve

# --- 1. ADIM: Veri Kümesini İndirme ve Galeri/Sorgu Olarak Ayırma ---
print("1. Olivetti Yüz Veri Kümesi İndiriliyor...")
faces = fetch_olivetti_faces(shuffle=False)
X = faces.data        # 400 görüntü x 4096 piksel (64x64)
y = faces.target      # Kişi etiketleri (0'dan 39'a kadar 40 kişi)

print(f"Toplam Görüntü Sayısı: {X.shape[0]}")
print(f"Her Bir Görüntünün Piksel Boyutu: {X.shape[1]} (64x64)")
print(f"Toplam Kişi Sayısı: {len(np.unique(y))}\n")

# Galeri (7 fotoğraf) ve Sorgu (3 fotoğraf) olarak ayırma
X_gallery, y_gallery = [], []
X_probe, y_probe = [], []

for person_id in range(40):
    person_indices = np.where(y == person_id)[0]
    
    # İlk 7'si Galeriye (Eğitim)
    X_gallery.append(X[person_indices[:7]])
    y_gallery.append(y[person_indices[:7]])
    
    # Kalan 3'ü Sorguya (Olay Yeri / Test)
    X_probe.append(X[person_indices[7:]])
    y_probe.append(y[person_indices[7:]])

X_gallery = np.vstack(X_gallery)
y_gallery = np.concatenate(y_gallery)
X_probe = np.vstack(X_probe)
y_probe = np.concatenate(y_probe)

print(f"✅ Galeri (Gallery) Seti Hazır: {X_gallery.shape[0]} fotoğraf")
print(f"✅ Sorgu (Probe) Seti Hazır: {X_probe.shape[0]} fotoğraf")


# --- 2. ADIM: Eigenfaces (PCA + SVM) Modelinin Eğitimi ---
print("\n--- 2. ADIM: Eigenfaces (PCA + SVM) Modelinin Eğitimi ---")

# PCA ile Özellik Boyutunu Düşürme (Temel Bileşen Sayısı: 50)
n_components = 50
pca = PCA(n_components=n_components, whiten=True, random_state=42)

# Sadece Galeri seti üzerinde PCA'i eğit
pca.fit(X_gallery)

# Hem Galeri hem de Sorgu setini vektör alanına izdüşür (transform)
X_gallery_pca = pca.transform(X_gallery)
X_probe_pca = pca.transform(X_probe)

print(f"Piksel Boyutu 4096'dan {n_components} Eigenface bileşenine düşürüldü.")

# Destek Vektör Makineleri (SVM) Sınıflandırıcısını Eğitme
clf = SVC(kernel='rbf', class_weight='balanced', C=1000.0, gamma=0.001, random_state=42)
clf.fit(X_gallery_pca, y_gallery)

# Sorgu Seti Üzerinde Tahmin Yapma ve Rank-1 Başarımı
y_pred = clf.predict(X_probe_pca)
rank1_acc = accuracy_score(y_probe, y_pred)

print(f"🎯 Eigenfaces (PCA + SVM) Rank-1 Accuracy: %{rank1_acc * 100:.2f}")


# --- 3. ADIM: Adli Doğrulama Metrikleri (FMR, FNMR, EER, LR) ---
print("\n--- 3. ADIM: Adli Doğrulama Metrikleri (FMR, FNMR, EER) ---")

sim_matrix = cosine_similarity(X_probe_pca, X_gallery_pca)
y_true_matrix = (y_probe[:, None] == y_gallery[None, :]).astype(int)

scores = sim_matrix.flatten()
labels = y_true_matrix.flatten()

fpr, tpr, thresholds = roc_curve(labels, scores)
fmr = fpr
fnmr = 1 - tpr

eer_idx = np.nanargmin(np.absolute(fmr - fnmr))
eer = fmr[eer_idx]
optimal_threshold = thresholds[eer_idx]

print(f"⚖️ Equal Error Rate (EER): %{eer * 100:.2f}")
print(f"🎯 EER için Eşik Değer (Threshold): {optimal_threshold:.4f}")

same_scores = scores[labels == 1]
diff_scores = scores[labels == 0]

sample_score = optimal_threshold
p_same = np.mean(same_scores >= sample_score)
p_diff = np.mean(diff_scores >= sample_score)

lr = p_same / (p_diff + 1e-9)
print(f"📊 Örnek Eşik Değeri ({sample_score:.2f}) için Adli Olabilirlik Oranı (LR): {lr:.2f}")


# --- 4. ADIM: Stres Testi (CCTV / Düşük Kalite Simülasyonu) ---
print("\n--- 4. ADIM: Stres Testi (CCTV / Düşük Kalite Simülasyonu) ---")

def degrade_to_cctv(images):
    degraded_images = []
    for img in images:
        img_2d = img.reshape(64, 64)
        low_res = cv2.resize(img_2d, (16, 16), interpolation=cv2.INTER_NEAREST)
        degraded = cv2.resize(low_res, (64, 64), interpolation=cv2.INTER_NEAREST)
        degraded = cv2.GaussianBlur(degraded, (5, 5), 0)
        noise = np.random.normal(0, 0.03, degraded.shape)
        degraded = np.clip(degraded + noise, 0, 1)
        degraded_images.append(degraded.flatten())
    return np.array(degraded_images)

X_probe_cctv = degrade_to_cctv(X_probe)
X_probe_cctv_pca = pca.transform(X_probe_cctv)

y_pred_cctv = clf.predict(X_probe_cctv_pca)
rank1_cctv_acc = accuracy_score(y_probe, y_pred_cctv)

print(f"📉 Orijinal Kalitede Rank-1 Accuracy: %{rank1_acc * 100:.2f}")
print(f"⚠️ CCTV Kalitesinde (Bozulmuş) Rank-1 Accuracy: %{rank1_cctv_acc * 100:.2f}")
print(f"🔴 Performans Kaybı (Drop): %{(rank1_acc - rank1_cctv_acc) * 100:.2f}")


# --- 5. ADIM: Görselleştirme ve Grafik Kaydı (Raporda Kullanmak İçin) ---
print("\n--- 5. ADIM: Orijinal vs CCTV Görselleştirmesi Hazırlanıyor ---")
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(X_probe[0].reshape(64, 64), cmap='gray')
plt.title("Orijinal Sorgu (Probe)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(X_probe_cctv[0].reshape(64, 64), cmap='gray')
plt.title("CCTV Kalitesi (Stres Testi)")
plt.axis('off')

plt.tight_layout()
plt.savefig("face_comparison.png")  # Klasöre resim olarak kaydeder
print("📸 Görsel karşılaştırma 'face_comparison.png' olarak kaydedildi.")
plt.show()