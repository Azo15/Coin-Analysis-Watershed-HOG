import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import os


def apply_watershed(image_path, tl, eur):
    # Resim yükleme kontrolü
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} bulunamadı!")
        return None

    # Görüntü işleme adımları (Örneğindeki akışa uygun)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Gürültü giderme

    # Otsu Thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfolojik işlemler ve Mesafe Dönüşümü
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Kesin ön plan (paralar) tespiti
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Marker oluşturma
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1

    # Watershed uygulama
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]  # Sınırları kırmızı boya

    # Sonuç üzerine miktar yazdır
    text = f"TL: {tl} | EUR: {eur}"
    cv2.putText(img, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
    return img


def apply_hog(image_path, tl, eur):
    img = cv2.imread(image_path)
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG Özellik Çıkarımı
    # orientations=8 dairesel objeler (paralar) için idealdir
    features, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True)

    # HOG görselini normalize et ve belirginleştir
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_output = (hog_image_rescaled * 255).astype(np.uint8)

    # DÜZELTİLEN SATIR: COLOR_GRAY2BGR (2 rakamı eklendi)
    hog_color = cv2.cvtColor(hog_output, cv2.COLOR_GRAY2BGR)

    # HOG sonucu üzerine miktar yazdır
    text = f"HOG - TL: {tl} | EUR: {eur}"
    cv2.putText(hog_color, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    return hog_color


# --- ANA PROGRAM AKIŞI ---

# Resimlerin olduğu klasörde 1.jpg, 2.jpg ve 3.jpg olmalı
isleme_listesi = [
    {"dosya": "1.jpg", "tl": "1.35", "eur": "1.70"},
    {"dosya": "2.jpg", "tl": "1.40", "eur": "1.60"},
    {"dosya": "3.jpg", "tl": "1.25", "eur": "1.55"}
]

print("Analiz baslatiliyor...")

for i, veri in enumerate(isleme_listesi, 1):
    dosya_adi = veri["dosya"]

    if os.path.exists(dosya_adi):
        print(f"Islem yapiliyor: {dosya_adi}")

        # Watershed İşlemi
        w_sonuc = apply_watershed(dosya_adi, veri["tl"], veri["eur"])
        if w_sonuc is not None:
            cv2.imwrite(f"sonuc_{i}_watershed.jpg", w_sonuc)

        # HOG İşlemi
        h_sonuc = apply_hog(dosya_adi, veri["tl"], veri["eur"])
        if h_sonuc is not None:
            cv2.imwrite(f"sonuc_{i}_hog.jpg", h_sonuc)
    else:
        print(f"Atlandi: {dosya_adi} bulunamadi.")

print("\nIslem Tamamlandi! Toplam 6 sonuc dosyasi kaydedildi.")