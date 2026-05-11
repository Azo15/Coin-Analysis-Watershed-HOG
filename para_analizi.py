import cv2
import numpy as np


def analyze_coins(image_path, i):
    img = cv2.imread(image_path)
    if img is None: return

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Hough Circle Transform ile paraları tespit et
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=150)

    total_tl = 0
    total_eur = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            center = (c[0], c[1])
            radius = c[2]

            # Basit bir mantık: Resimdeki renk dağılımına göre ayrım (Örnek eşikleme)
            # Not: Gerçek projede burada her daire içindeki renk ortalamasına bakılır
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, center, radius, 255, -1)
            mean_val = cv2.mean(img, mask=mask)[:3]

            # Renk analizi ile TL ve Euro ayrımı (Temsili Renkler)
            # Euro genellikle daha sarımsı/altın tonludur (BGR formatında R ve G yüksektir)
            if mean_val[2] > 160:  # Kırmızı/Sarı tonu baskınsa Euro varsayalım
                color = (0, 255, 255)  # Sarı Çember (Euro)
                label = "EUR"
                total_eur += 0.50  # Örnek birim değer
            else:
                color = (255, 0, 0)  # Mavi Çember (TL)
                label = "TL"
                total_tl += 1.0  # Örnek birim değer

            # Çemberi çiz ve etiketi yaz
            cv2.circle(output, center, radius, color, 3)
            cv2.putText(output, label, (center[0] - 15, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # En üste toplam sonuçları yazdır
    header_text = f"TOPLAM -> TL: {total_tl:.2f} | EUR: {total_eur:.2f}"
    cv2.rectangle(output, (0, 0), (output.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(output, header_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imwrite(f"guncel_sonuc_{i}.jpg", output)
    print(f"Resim {i} kaydedildi.")


# Çalıştır
for i in range(1, 4):
    analyze_coins(f"{i}.jpg", i)