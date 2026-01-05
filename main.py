import cv2
import numpy as np
from scipy.io import wavfile

# --- AYARLAR ---
VIDEO_PATH = "track_deneme_car.mp4"   # Video dosyanın adı (Aynı klasörde olsun)
AUDIO_PATH = "track_deneme_ses.wav"     # İşlenecek ses dosyanın adı (WAV formatında olmalı)
OUTPUT_PATH = "cikti_car.wav"  # Oluşacak yeni ses dosyası

def normalize_position(x, width):

    # 0 ile 1 arasına getir
    normalized = x / width
    # -1 ile +1 arasına genişlet (Panlama için)
    # Formül: (değer * 2) - 1
    pan_value = (normalized * 2) - 1
    
    # Değerleri -1 ve 1 sınırlarında tut (Taşma olmasın)
    return np.clip(pan_value, -1.0, 1.0)

def main():
    print("Video analiz ediliyor...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Hata: Video dosyası açılamadı!")
        return

    # İlk kareyi oku
    ret, frame = cap.read()
    if not ret:
        print("Video okunamadı!")
        return

    print("Lütfen pencerede takip edilecek nesneyi seçin ve ENTER'a basın.")
    # selectROI(PencereAdı, Görüntü) -> (x, y, w, h) döner
    bbox = cv2.selectROI("Takip Hedefi Secimi", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Takip Hedefi Secimi")

    # Tracker başlat 
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    frame_pan_values = []
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Takipçiyi güncelle
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = box
            center_x = x + (w / 2) # Nesnenin orta noktasını al
            
            # Görselleştirme
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            
            pan_val = normalize_position(center_x, video_width)
            frame_pan_values.append(pan_val)
        else:
            if frame_pan_values:
                frame_pan_values.append(frame_pan_values[-1])
            else:
                frame_pan_values.append(0.0)

        # Commentlersen işlem hızlanır
        cv2.imshow("Takip Ediliyor...", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC ile çık
            break
        
    cap.release()
    cv2.destroyAllWindows()
    print(f"Takip tamamlandı. Toplam {len(frame_pan_values)} kare işlendi.")

    # 2. SES İŞLEME VE SENKRONİZASYON
    print("Ses işleniyor...")
    
    # Ses dosyasını oku (Scipy ile)
    sample_rate, data = wavfile.read(AUDIO_PATH)
    print(f"Ses yüklendi - Sample rate: {sample_rate}, Shape: {data.shape}, Dtype: {data.dtype}")
    print(f"Ses değer aralığı: min={data.min()}, max={data.max()}")
    
    # Eğer ses mono ise (tek kanal), stereoya çevirmemiz lazım
    if len(data.shape) == 1:
        data = np.stack((data, data), axis=1) # (N, 2) yapar
        
    total_samples = data.shape[0]
    
    # Pan verisini (Video FPS) ses verisine (44100 Hz) eşitlemek (INTERPOLATION)
    # Elimizde örneğin 1000 tane video karesi verisi var ama 5.000.000 tane ses verisi var.
    # Aradaki boşlukları numpy ile doldurarak pürüzsüz bir eğri oluşturacağız.
    
    x_video = np.linspace(0, total_samples, len(frame_pan_values))
    x_audio = np.arange(total_samples)
    
    # Lineer interpolasyon ile her ses örneği için bir pan değeri üret
    smooth_pan = np.interp(x_audio, x_video, frame_pan_values)
    print(f"Pan değerleri - min={smooth_pan.min():.3f}, max={smooth_pan.max():.3f}, ortalama={smooth_pan.mean():.3f}")
    
    # Pan Yasası (Constant Power Panning) Uygulama
    # Pan değeri -1 (Sol) ile +1 (Sağ) arasında.
    # Bunu 0 ile PI/2 (90 derece) arasına çevirip sin/cos kullanacağız.
    
    # -1 -> 0 radyan (Sadece Sol)
    #  0 -> PI/4 radyan (Ortada Eşit)
    # +1 -> PI/2 radyan (Sadece Sağ)
    
    theta = (smooth_pan + 1) / 2 * (np.pi / 2)
    
    left_gain = np.cos(theta)
    right_gain = np.sin(theta)
    print(f"Gain değerleri - left_gain: min={left_gain.min():.3f}, max={left_gain.max():.3f}")
    print(f"Gain değerleri - right_gain: min={right_gain.min():.3f}, max={right_gain.max():.3f}")
    
    # Orijinal sesi float'a çevir (işlem yapabilmek için)
    audio_float = data.astype(np.float32)
    
    # Sol ve Sağ kanalları hesapla
    processed_audio = np.zeros_like(audio_float)
    mono = (audio_float[:, 0] + audio_float[:, 1]) / 2  # Mix to mono
    processed_audio[:, 0] = mono * left_gain  # Panned mono to left
    processed_audio[:, 1] = mono * right_gain # Panned mono to right
    
    print(f"İşlenmiş ses - min={processed_audio.min():.1f}, max={processed_audio.max():.1f}")
    
    # Tekrar 16-bit integer formatına çevirip kaydet
    # Float32 (-1.0 to 1.0) değerlerini int16 (-32768 to 32767) aralığına ölçeklendir
    final_audio = np.int16(processed_audio * 32767)
    print(f"Final ses (int16) - min={final_audio.min()}, max={final_audio.max()}")
    wavfile.write(OUTPUT_PATH, sample_rate, final_audio)
    
    print(f"İşlem bitti! Dosya kaydedildi: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()