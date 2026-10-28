# GitHub teslim kontrolü

## Kalan kişisel adımlar

- Görselden okunan `22120606015` numarasının doğru olduğunu ve `labels.csv` etiketlerini kontrol edin.
- Tamamlanan raporu okuyun; deneyleri açıklayabildiğinizden emin olun.
- Projeyi kendi GitHub hesabınızda depoya yükleyin; öğretmenin erişebildiğinden emin olun.

## Tarayıcı üzerinden yükleme

1. GitHub hesabınıza giriş yapın, yeni depo oluşturun. Önerilen ad: `mnist-mlp-handwritten-digits`.
2. ZIP'i bilgisayarınızda açın. ZIP dosyasının kendisini değil, içindeki `mnist-mlp` klasörünün dosyalarını yükleyin. README deponun kökünde olmalı.
3. `Add file → Upload files` ile kodu, yapılandırmaları, notebook'u, raporu, sonuçları ve gerçek el yazısı klasörünü ekleyin. Eğitim veri dosyasını, sanal ortamı veya geçici dosyaları eklemeyin.
4. Değişiklikleri kaydedin. README'deki grafik ve bağlantıları kontrol edin.
5. Öğretmenin istediği erişim biçimini sağlayın ve depo bağlantısını teslim edin.

## Git kullananlar için

Proje kökünde, boş bir GitHub deposu oluşturduktan sonra:

```bash
git init
git add .
git commit -m "Add MNIST MLP experiments and handwriting evaluation"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADI/mnist-mlp-handwritten-digits.git
git push -u origin main
```

`KULLANICI_ADI` kısmını kendi hesabınızla değiştirin. Bu komutlar henüz çalıştırılmamıştır; bu paket GitHub'da yayımlanmış bir depo değildir.

## Kontrol listesi

- [x] En az iki gizli katman ve ReLU.
- [x] Sıfırdan eğitim; CNN/hazır model yok.
- [x] Train/validation/test ayrımı ve kaydedilmiş indeksler.
- [x] Loss, accuracy, test accuracy ve eğitim süresi.
- [x] Eğitim grafikleri ve hiperparametre karşılaştırmaları.
- [x] Sabit seçim kuralı; test yalnızca seçimden sonra.
- [x] Çalıştırma yönergesi, bağımlılıklar ve kontrol testleri.
- [x] Gerçek öğrenci numarası fotoğrafı ve rakam kırpımları.
- [x] El yazısı tahmin sonuçları ve gözleme dayalı yorum.
- [x] README/rapordaki bekleyen bölüm notlarının güncellenmesi.
- [ ] Öğrencinin kodu ve sonuçları açıklayabilmesi.
- [ ] GitHub deposu ve teslim bağlantısı.
