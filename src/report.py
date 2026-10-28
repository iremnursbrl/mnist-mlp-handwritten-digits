"""Generate a Turkish report only from completed experiment records."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate(root):
    root=Path(root)
    rows=json.loads((root/'results.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    cfg=json.loads((root/'config.json').read_text())
    env=json.loads((root/'environment.json').read_text())
    groups={e['name']:[r for r in rows if r['experiment']==e['name']] for e in cfg['experiments']}
    def summary(rs,key,percent=False):
        a=np.array([r[key] for r in rs])*(100 if percent else 1)
        return f'{a.mean():.3f} ± {a.std(ddof=1):.3f}'
    lines=['# MNIST ile MLP tabanlı el yazısı rakam tanıma',
           '**Öğrenci:** İrem Nur Sabırlı  \n**Durum:** MNIST deneyleri tamamlandı; öğrencinin kendi el yazısı deneyi görseller bekleniyor.',
           '## 1. Amaç ve deney tasarımı',
           'Amaç, 28×28 gri tonlamalı görüntüyü 784 özellikli vektöre dönüştüren MLP ile 0–9 sınıflarını tanımak ve yapılandırma değişikliklerini kontrollü karşılaştırmaktır. CNN veya hazır eğitilmiş model kullanılmamıştır. Her deney ve başlangıç için ağırlıklar sıfırdan oluşturulmuştur.',
           'MNIST’in resmî 60.000 eğitim örneği, sınıf oranları korunarak 50.000 train ve 10.000 validation örneğine ayrılmıştır. Resmî 10.000 test örneği ayrı tutulmuştur. Bölme tohumu 42’dir; indeksler `split_indices.npz` dosyasındadır. Tüm modeller aynı bölünmeyi kullanır. Piksel değerleri 255’e bölünerek [0,1] aralığına alınır; veri üzerinden ortalama veya standart sapma öğrenilmez.',
           '## 2. Hiperparametreler',
           '| Deney | Katmanlar | Dropout | Adam öğrenme oranı | Parametre |\n|---|---|---:|---:|---:|']
    for e in cfg['experiments']:
        lines.append(f"| {e['name']} | {' → '.join(map(str,[784]+e['hidden']+[10]))} | {e['dropout']} | {e['lr']} | {groups[e['name']][0]['parameters']:,} |")
    lines += [f"Tüm deneylerde ReLU, CrossEntropyLoss, Adam (betas=(0.9, 0.999), eps=1e-8, weight_decay=0), batch_size={cfg['batch_size']}, {cfg['epochs']} epoch ve başlangıç tohumları {cfg['seeds']} kullanılmıştır. Scheduler, veri artırma, batch normalization veya erken durdurma uygulanmamıştır. Dropout yalnızca ilgili deneyde her gizli katmandaki ReLU’dan sonra eklenmiştir.",
       'Çıkış katmanı 10 ham skor (logit) üretir. CrossEntropyLoss log-softmax işlemini içerdiği için eğitimde ayrıca softmax uygulanmaz; el yazısı tahmininde skorları göstermek için softmax kullanılır.',
       'Her koşuda en düşük validation loss değerine sahip epoch kaydedilir; eşitlikte ilk epoch korunur. Yapılandırma, üç koşunun bu validation loss değerlerinin ortalaması en düşük olanıdır. Sonra seçilen yapılandırmanın validation loss değeri en düşük tohumu el yazısı için temsilci model olarak alınır. Bu karar `selection.json` dosyasına test dizileri okunmadan önce yazılır. Tüm dondurulmuş modellerin test sonuçları yalnızca nihai karşılaştırma için bir kez hesaplanır; test üzerinden yeni ayar seçilmez.',
       '## 3. Ölçüm tanımları',
       'Accuracy, doğru sınıflandırılan örneklerin toplam örneklere oranıdır. Loss, doğru sınıfa verilen olasılığa ilişkin çapraz entropi kaybıdır; daha düşük olması tercih edilir. Her epoch sonunda train ve validation ölçümleri aynı ağırlıklarla eval modunda yeniden hesaplanır; dropout ölçüm sırasında kapalıdır. Kayıplar örnek sayısıyla ağırlıklandırılmış ortalamadır.',
       'Tablodaki train/validation değerleri son epoch değil, ilgili koşunun en iyi validation loss epoch’una aittir. Eğitim süresi optimizasyonu, epoch sonu train/validation ölçümlerini ve checkpoint yazımını kapsar; veri indirme, grafik çizimi ve test değerlendirmesini kapsamaz. Süreler aynı CPU üzerinde ardışık koşulardan gelir ve donanıma bağlıdır.',
       '## 4. Sonuçlar',
       'Değerler üç başlangıcın ortalaması ± örnek standart sapmasıdır (ddof=1). Accuracy yüzde olarak verilmiştir. Standart sapma güven aralığı değildir.',
       '| Deney | Train loss | Val loss | Train acc % | Val acc % | Test acc % | Süre (s) |\n|---|---:|---:|---:|---:|---:|---:|']
    for name,rs in groups.items():
        cells=[summary(rs,'train_loss'),summary(rs,'validation_loss'),summary(rs,'train_accuracy',True),summary(rs,'validation_accuracy',True),summary(rs,'test_accuracy',True),summary(rs,'training_seconds')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['### Tüm koşular', '| Deney | Seed | En iyi epoch | Train loss | Val loss | Train acc % | Val acc % | Test acc % | Süre (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['seed']} | {r['best_epoch']} | {r['train_loss']:.4f} | {r['validation_loss']:.4f} | {100*r['train_accuracy']:.2f} | {100*r['validation_accuracy']:.2f} | {100*r['test_accuracy']:.2f} | {r['training_seconds']:.2f} |")
    representative=next(r for r in rows if r['experiment']==selection['experiment'] and r['seed']==selection['seed'])
    lines += ['## 5. Model seçimi ve yorum',
       f"Validation ölçütüne göre **{selection['experiment']}** seçilmiştir. El yazısı için seçilen checkpoint `{selection['checkpoint']}` dosyasıdır. Bu modelin test doğruluğu **%{100*representative['test_accuracy']:.2f}**, validation doğruluğu **%{100*representative['validation_accuracy']:.2f}** ve en iyi epoch’u **{representative['best_epoch']}** olmuştur."]
    baseline=groups['baseline']
    for name in groups:
        if name=='baseline': continue
        rs=groups[name]
        valdelta=np.mean([r['validation_accuracy'] for r in rs])-np.mean([r['validation_accuracy'] for r in baseline])
        lossdelta=np.mean([r['validation_loss'] for r in rs])-np.mean([r['validation_loss'] for r in baseline])
        lines.append(f"**{name}:** Temel modele göre ortalama validation accuracy farkı {100*valdelta:+.3f} yüzde puanı, validation loss farkı {lossdelta:+.4f} olmuştur. Değiştirilen faktör bu karşılaştırmada tek başına tutulmuştur; sonuç yalnızca kullanılan hiperparametreler ve 12 epoch bütçesi için yorumlanmalıdır.")
    lines += ['Geniş ağ daha fazla temsil kapasitesine ve hesaplama maliyetine sahiptir. Dropout eğitimde bazı aktivasyonları rastgele sıfırlayarak düzenlileştirme sağlar; her koşulda daha yüksek accuracy garantilemez. Daha düşük öğrenme oranı aynı epoch bütçesinde daha yavaş yakınsayabilir. Accuracy ve loss aynı yönde değişmek zorunda değildir: doğru/yanlış sınıf kararı değişmeden modelin olasılıkları değişebilir.',
       'Üç başlangıç, başlangıç ağırlıkları ve mini-batch sıralamasından kaynaklanan değişkenliği görünür kılar. Veri bölünmesi sabittir; bu deney çapraz doğrulama değildir ve üç koşu istatistiksel üstünlük iddiası için yeterli değildir. Test sonuçları görüldükten sonra yeni yapılandırma denenerek test kümesine dolaylı uyum sağlanmamıştır.',
       '### Öğrenme eğrileri',
       'Aşağıda her yapılandırmanın seed=42 eğrileri gösterilir. Diğer iki tohumun eğrileri ilgili klasörlerdedir. Dikey çizgi, checkpoint seçiminde kullanılan epoch’u gösterir. Train loss düşerken validation loss artması aşırı öğrenme işaretidir; seçilen checkpoint bu nedenle son epoch’tan daha erken olabilir.']
    for name in groups: lines.append(f'![{name} eğrileri]({name}_seed42/curves.png)')
    lines += ['### Seçilen modelin hataları','![Karışıklık matrisi](confusion_matrix.png)',
              'Karışıklık matrisinde satırlar gerçek, sütunlar tahmin edilen sınıflardır. Köşegen dışındaki sayılar hatalardır. Sınıf bazlı precision, recall ve F1 değerleri `classification_report.json` dosyasındadır.',
              '![İlk 24 yanlış test tahmini](test_errors.png)',
              'Görseller test sırasındaki ilk 24 yanlış örnektir; rastgele veya tüm hataları temsil eden bir örneklem değildir. İnceleme yalnızca hata analizi içindir; bu örneklere bakılarak model yeniden ayarlanmamıştır.',
              '## 6. Kendi el yazısı — henüz tamamlanmadı',
              'Öğrencinin kendi okul numarası görseli henüz sağlanmamıştır. Bu nedenle bu bölüm için başarı oranı, tahmin veya dağılım farkına ait ölçüm raporlanamaz. `handwriting/README.md` yönergesi izlenerek orijinal görsel ve sıralı rakam kırpımları eklenmelidir. Tahmin aracı, gri tonlama, arka plan yönü düzeltme, rakam sınırlarını kırpma, oranı koruyarak 20×20 içine sığdırma, 28×28 tuvale yerleştirme, kütle merkezi hizalama ve [0,1] normalizasyonunu uygular.',
              'Görsel geldikten sonra rakam bazlı doğruluk, numaranın tamamının eşleşmesi ve hatalı rakamlar raporlanmalıdır. MNIST ile fark; kalem kalınlığı, kişisel yazım biçimi, fotoğraf ışığı ve hizalama gibi olası etkenlerle, gerçekten gözlenen örnekler üzerinden tartışılmalıdır. Tek numara genel genelleme başarısı için yeterli değildir. Softmax skoru kalibre edilmiş güven olasılığı olarak yorumlanmamalıdır.',
              '## 7. Yeniden üretilebilirlik',
              f"Python {env['python']}; PyTorch {env['torch']}; NumPy {env['numpy']}; aygıt: {env['device']}; thread sayısı: {env['threads']}. Ortamın ayrıntısı `environment.json` dosyasındadır. Aynı ortamda deterministik işlemler etkinleştirilmiştir; farklı donanım/kütüphane sürümleri birebir aynı sonuçları garanti etmez.",
              f"MNIST dosyası SHA-256: `{env['mnist_sha256']}`.",
              'Ham ölçümler `results.json` ve `results.csv` dosyalarında, epoch geçmişleri alt klasörlerde bulunur. Rapor bu kayıtlar üzerinden `python -m src.report --results results/main` komutuyla üretilir.',
              '## 8. Kaynaklar',
              '- [MNIST — TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mnist)\n- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)\n- [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)\n- [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)',
              '## 9. Genel değerlendirme',
              f"Tam bağlantılı MLP ile MNIST sınıflandırması gerçekleştirilmiş, tüm modeller sıfırdan eğitilmiş ve model seçimi validation verisiyle sınırlandırılmıştır. Bu deney düzeninde seçilen {selection['experiment']} yapılandırmasının üç koşudaki ortalama test doğruluğu %{100*np.mean([r['test_accuracy'] for r in groups[selection['experiment']]]):.3f} olmuştur. Kendi el yazısı değerlendirmesi tamamlanmadan ödevin bütün gereksinimlerinin karşılandığı söylenemez."]
    handwriting_path = root.parent/'handwriting/predictions.json'
    if handwriting_path.exists():
        hand = json.loads(handwriting_path.read_text())
        same_checkpoint = Path(hand['checkpoint']).resolve() == (root/selection['checkpoint']).resolve()
        if same_checkpoint:
            start = lines.index('## 6. Kendi el yazısı — henüz tamamlanmadı')
            end = lines.index('## 7. Yeniden üretilebilirlik')
            correct = sum(d['label'] == d['prediction'] for d in hand['digits'])
            section = [
                '## 6. Kendi el yazısı değerlendirmesi',
                'Öğrencinin sağladığı özgün fotoğraf `handwriting/raw/student_number.jpg` dosyasında korunmuştur. Görseldeki numara insan tarafından 22120606015 olarak okunmuş ve gerçek etiketler bu okumadan alınmıştır. Model tahmini etiket oluşturmak için kullanılmamıştır.',
                'Orijinal 1550×779 fotoğraftan soldan sağa 11 dikdörtgen kırpım hazırlanmıştır. Koordinatlar `handwriting/crop_metadata.json` içinde, tekrar üretim kodu `src/prepare_handwriting.py` dosyasındadır. Kırpımlar model çıktıları görülmeden belirlenmiştir. İkinci ve üçüncü kırpımda komşu rakama ait küçük izler görülebilir; bunlar sonuç sonrasında temizlenerek yeniden deneme yapılmamıştır.',
                'Her kırpıma mevcut ön işleme uygulanmıştır: gri tonlama, açık arka planın ters çevrilmesi, arka plan seviyesinin bastırılması, rakam sınırlarının bulunması, oranı korunarak 20×20 alana sığdırma, 28×28 siyah tuvale yerleştirme, kütle merkezine hizalama ve piksel/255 normalizasyonu. Çıktılar `results/handwriting/digit_*_28x28.png` dosyalarındadır.',
                f"Görselden okunan numara: **{hand['true_number']}**. Model tahmini: **{hand['predicted_number']}**. Doğru rakam sayısı: **{correct}/{hand['n_digits']}**. Rakam doğruluğu: **%{100*hand['digit_accuracy']:.2f}**. Tüm numaranın eşleşmesi: **{'Evet' if hand['exact_number_match'] else 'Hayır'}**.",
                '![El yazısı tahminleri](../handwriting/predictions.png)',
                '| Konum | Gerçek | Tahmin | Softmax skoru (%) |\n|---|---:|---:|---:|',
            ]
            for i,d in enumerate(hand['digits'],1):
                section.append(f"| {i} | {d['label']} | {d['prediction']} | {100*d['softmax_score']:.2f} |")
            section.extend([
                'Softmax skorları kalibre edilmiş güven olasılığı değildir. En düşük tahmin skoru ikinci rakamda yaklaşık %87,59’dur; bu rakam da doğru tanınmıştır. Görsel farklılıklar veya komşu rakam izi buna katkıda bulunabilir, ancak tek örnek üzerinden neden-sonuç çıkarımı yapılamaz.',
                'MNIST test doğruluğu %97,86 iken bu fotoğrafta 11/11 doğru tahmin elde edilmiştir. El yazısı örneklemi çok küçük ve dengesizdir: yalnızca 0, 1, 2, 5 ve 6 sınıfları vardır; 3, 4, 7, 8 ve 9 sınanmamıştır. Bu nedenle %100 sonucu, modelin gerçek dünyada MNIST’ten daha başarılı olduğunun kanıtı değildir.',
                'Fotoğrafta mavi/mor mürekkep, sıcak tonlu kâğıt, aydınlatma ve kişisel yazım biçimi farklılıkları görülür. Ön işleme bu girdiyi MNIST biçimine yaklaştırmıştır. Bu örnekte dağılım farkı yanlış sınıflandırmaya yol açmamıştır; buna rağmen genel dayanıklılık için daha fazla kişinin farklı koşullarda yazdığı, tüm sınıfları kapsayan bağımsız örnekler gerekir.',
                'Model ve hiperparametreler el yazısı sonucu görüldükten sonra değiştirilmemiştir. Görsel eğitimde veya model seçiminde kullanılmamıştır. Bu sonuç, elle ayrılmış rakamların sınıflandırılmasıdır; otomatik tam sayfa OCR veya otomatik numara bölme başarısı değildir.',
            ])
            lines[start:end] = section
            lines[1] = '**Öğrenci:** İrem Nur Sabırlı  \n**Öğrenci numarası (görselden okunan):** 22120606015  \n**Durum:** MNIST ve gerçek el yazısı deneyleri tamamlandı; GitHub yayımlama adımı bekleniyor.'
            lines[-1] = lines[-1].replace('Kendi el yazısı değerlendirmesi tamamlanmadan ödevin bütün gereksinimlerinin karşılandığı söylenemez.', 'Öğrencinin sağladığı numara fotoğrafındaki 11 rakamın tamamı doğru tanınmıştır. Sonuç tek, küçük örnekleme özgüdür; tüm sınıflarda veya farklı kişilerin el yazılarında genel başarı garantisi değildir.')
    text = re.sub(r'(?<=\|)\n\n(?=\|)', '\n', '\n\n'.join(lines))
    (root/'RAPOR.md').write_text(text,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=list(groups)
    for ax,key,title in [(axes[0],'validation_accuracy','Validation accuracy (%)'),(axes[1],'test_accuracy','Test accuracy (%)')]:
        means=[100*np.mean([r[key] for r in groups[n]]) for n in names]
        std=[100*np.std([r[key] for r in groups[n]],ddof=1) for n in names]
        ax.bar(names,means,yerr=std,capsize=5,color=['#455a64','#00796b','#5e35b1','#ef6c00'])
        ax.set_ylim(0,100); ax.set_title(title); ax.set_ylabel('Mean ± sample SD'); ax.tick_params(axis='x',rotation=20)
    fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=160); plt.close(fig)
    return representative

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',default='results/main')
    print(generate(p.parse_args().results))
