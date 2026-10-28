# Projeyi anlayarak sunma

## 90 saniyelik anlatımın iskeleti

MNIST'teki 28×28 görüntüleri 784 uzunluklu vektöre çevirdim. İki gizli katmanlı ve ReLU aktivasyonlu MLP'yi sıfırdan eğittim. Eğitim verisinin 10.000 örneğini validation için ayırdım; resmî test verisini model seçiminde kullanmadım. Temel modele karşı genişlik, dropout ve öğrenme oranı değişikliklerini ayrı ayrı denedim. Her yapılandırmayı üç farklı başlangıçla çalıştırdım. En iyi modeli validation loss'a göre seçtim; ardından test doğruluğunu ölçtüm. Kendi numaramın fotoğrafını rakamlara ayırıp aynı normalizasyonla modele verdim. 11 rakamın tamamı doğru tanındı; ancak tek numara ve beş farklı rakam sınıfı genel başarıyı ölçmek için yeterli değildir.

Bu iskeleti kendi anlayışınız ve yaptığınız işlemlerle uyumlu olarak anlatın. Kodun ve raporun nasıl hazırlandığını dersin kurallarına uygun, doğru biçimde açıklayın.

## Sorular ve yanıtlar

**Neden 784 giriş?** 28×28=784 gri seviye piksel vardır. Flatten yalnızca şekli değiştirir; öğrenilen parametresi yoktur.

**İki gizli katman nerede?** Temel modelde 128 ve 64 nöronlu katmanlardır. Giriş ve 10 nöronlu çıkış gizli katman sayılmaz.

**Parametre sayısı nasıl hesaplanır?** Bias dahil (784+1)×128 + (128+1)×64 + (64+1)×10 = 109.386.

**ReLU ne yapar?** max(0,x) uygular. Doğrusal katmanlar arasına doğrusal olmayan dönüşüm ekler; aksi halde katmanlar tek doğrusal dönüşüme indirgenebilir.

**Neden 10 çıkış?** Her çıkış bir rakam sınıfının skorudur. En yüksek skorun indeksi tahmindir.

**Neden CrossEntropyLoss?** Sınıflar birbirini dışlayan on rakamdır. Kayıp doğru sınıfın olasılığını artırmayı hedefler. PyTorch ham logit bekler; öncesinde softmax uygulamak gerekmez.

**Backpropagation ve Adam farklı mı?** Backpropagation kaybın parametrelere göre gradyanını hesaplar. Adam bu gradyanları ve geçmiş istatistiklerini kullanarak ağırlıkları günceller.

**Epoch ve batch nedir?** Epoch eğitim kümesinden bir geçiştir. Batch, bir ağırlık güncellemesinde kullanılan örnek grubudur. 50.000 örnek ve batch 128 için epoch başına 391 güncelleme vardır; son batch daha küçüktür.

**Train, validation ve test neden ayrı?** Train ağırlıkları öğrenir; validation model/epoch seçer; test seçilmiş sistemin görülmemiş verideki nihai performansını ölçer. Test sonucuna göre ayar değiştirmek test bilgisini geliştirmeye sızdırır.

**Neden stratify?** Eğitim ve validation kümelerinde rakam sınıflarının oranlarını yaklaşık korur.

**Neden üç seed?** Başlangıç ağırlıkları, veri sırası ve dropout rastgeleliği sonucu etkiler. Ortalama ve standart sapma bu etkiyi gösterir; üç deneme tüm olası değişkenliği ölçmez.

**En iyi accuracy ile en düşük loss aynı mı?** Hayır. Accuracy yalnızca en yüksek skorlu sınıfa bakar. Loss olasılıkların ne kadar doğru veya yanlış olduğuna da duyarlıdır. Seçim kuralı baştan validation loss olarak belirlenmiştir.

**Overfitting nasıl anlaşılır?** Train loss iyileşirken validation loss kötüleşebilir. Son epoch'u otomatik kullanmak yerine en düşük validation loss checkpoint'i kaydedilir. Bu projede erken durdurma yoktur; tüm 12 epoch çalışır.

**Dropout neden train/val grafiğini bozmaz?** Ölçümler eval modunda, dropout kapalıyken alınır. Eğitim güncellemelerinde dropout aktiftir.

**Numaranın tamamı mı, her rakam mı sınıflandırılıyor?** Model tek rakam sınıflandırır. Numaranın rakamları sırayla ayrı kırpılır, ayrı tahmin edilir ve birleştirilir. Bu bir metin tanıma/segmentasyon modeli değildir.

**MNIST yüksekken kendi el yazım neden yanlış çıkabilir?** Kalem kalınlığı, biçim, ışık, açı, boyut ve hizalama farklı olabilir. Bu distribution shift'tir. Ayrıca hatalı kırpma bir ön işleme sorunudur. Gözlem yapmadan belirli bir neden kesin ilan edilemez.

**%99 softmax skoru kesin doğru demek mi?** Hayır. Özellikle eğitim dağılımından farklı girdilerde model yüksek skorla yanılabilir.

## Sunum öncesi pratik

1. `core.py` içindeki ağı kendi sözlerinizle açıklayın.
2. `training` ile `eval` modları arasındaki farkı gösterin.
3. `selection.json` dosyasından seçilen modeli okuyun.
4. Rapordan iki yapılandırmayı sayısal olarak karşılaştırın.
5. Karışıklık matrisinde bir köşegen dışı hücreyi yorumlayın.
6. Kendi numaranızdaki yanlış tahminleri, varsa, ön işleme görüntüleriyle birlikte gösterin.
