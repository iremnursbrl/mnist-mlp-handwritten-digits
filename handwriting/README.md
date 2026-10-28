# Öğrencinin gerçek el yazısı görselleri

Öğrencinin gönderdiği fotoğraf `raw/student_number.jpg` olarak aynen korunmuştur. Görselden okunan numara **22120606015**tir. 11 rakamın özgün kırpımları `digits/` klasöründe, gerçek etiketleri `labels.csv`, kırpma koordinatları `crop_metadata.json` dosyasındadır. Etiketler model çıktısından değil görselin insan tarafından okunmasından elde edilmiştir.

Kırpımları tekrar oluşturma: `python -m src.prepare_handwriting`. Görsel boyutu 1550×779 olmalıdır. Bu komut yalnızca bu fotoğrafın kayıtlı koordinatlarını kullanır, otomatik segmentasyon yapmaz. Model değiştirilmeden yapılan ilk değerlendirmede 11/11 doğru tahmin alınmıştır.

## Başka bir fotoğraf hazırlanacaksa

1. Beyaz, çizgisiz kâğıda koyu kalemle okul numaranızı yazın. Rakamlar birbirine değmesin. Kâğıda çizilmiş kutu/çizgi, imza veya başka yazılar eklemeyin.
2. Fotoğrafı tepeden, gölgesiz ve net çekin. Orijinali `raw/student_number.jpg` olarak saklayın.
3. Her rakamı çevresinde küçük bir beyaz payla ayrı kırpın. Sırayla `digits/01.png`, `digits/02.png` vb. kaydedin. Tekrarlanan rakamları da ayrı tutun. Yeniden çizim veya yapay rakam üretimi kullanmayın.
4. `labels.csv` dosyasına sıra ile dosya yolu ve gerçek etiketi ekleyin. Örnek satır biçimi: `digits/01.png,2` (yalnızca ilk rakamınız gerçekten 2 ise).
5. Görselleri 28×28'e elle sıkıştırmayın; kod oranı koruyarak ölçekler, kütle merkezine taşır ve 255'e bölerek normalize eder.
6. README'deki tahmin komutunu çalıştırın. `predictions.png` ve tekil 28×28 çıktıları kontrol edin. Yanlış bölünmüş bir rakamın tahmini model hatasından ayrı değerlendirilmelidir.

Otomatik numara bölme özellikle bitişik rakamlarda güvenilir olmadığı için açık ve denetlenebilir tek-rakam kırpımları kullanılır. Model doğru rakamları tahmin etsin diye etiketlerden yararlanarak görüntüler değiştirilmemelidir.

## Değerlendirme sınırı

Tek okul numarası küçük, sınıfları dengeli olmayan bir örneklemdir. Rakam doğruluğunu ve tüm numaranın birebir eşleşmesini birlikte raporlayın. Eğitimde kullanılmayan kalem kalınlığı, ışık, açı, yazı biçimi ve arka plan farklılıkları distribution shift oluşturabilir.
