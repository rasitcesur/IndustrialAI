# CV Atlas MIL

**Döküm yüzeylerinde bölgesel olarak eşleştirilmiş kanıtlara dayalı görüntü sınıflandırması**

CV Atlas MIL, sabit görüntü özelliklerini, sağlam ürün referans atlasını, Attention-MIL yaklaşımını ve Extra Trees sınıflandırıcılarını birleştiren araştırma amaçlı bir görsel kalite kontrol sistemidir.

Modelin temel yaklaşımı, **gri-seviye/kontrast, morfoloji ve atlas farklılıklarının aynı görüntü bölgesinde birlikte değerlendirilmesi** ve daha sonra görüntü düzeyindeki kararın verilmesidir. Eğitim için yalnızca görüntü etiketleri (`defective` / `ok`) kullanılır; piksel düzeyinde kusur maskesi zorunlu değildir.

**Durum:** Araştırma prototipi. Bağımsız üretim partisi ve kamera koşullarıyla doğrulanmış bir endüstriyel kontrol ürünü değildir.

## Python paketi

Uygulama tek bir `cv_atlas_mil/` Python paketi altında sunulur. Eğitilmiş modeller yerel `cv_atlas_mil` sınıf yollarıyla kaydedilir. Eski dahili modül yollarıyla kaydedilmiş modellerin kullanılmadan önce dönüştürülmesi veya yeniden eğitilmesi gerekir.

## Mimari

| Uzman | İşlev | Füzyon ağırlığı |
|---|---|---:|
| Extra Trees | Sabit konvolüsyonlardan elde edilen 128 özellik | %45 |
| Sağlam referans atlası | 64 bölgenin normal örüntülerden sapması | %10 |
| Önceki Atlas-MIL | Bölgesel geometri ve atlas kanıtı | %5 |
| **Bölgesel Ortak MIL** | **Aynı bölgeden yoğunluk/geometri (14), morfoloji (28), atlas (4), koordinat (2)** | **%5** |
| Yüksek çözünürlüklü morfoloji | 16/32/64 piksellik çizgi, gradyan, black-hat/top-hat özellikleri | %35 |

Pozitif sınıfın anlamı `defective=0` etiketidir. Nihai kusur skoru:

\[
S=0.45S_{ET}+0.10S_{Atlas}+0.05S_{MIL}+0.05S_{Regional}+0.35S_{Morph}.
\]

`S >= 0.5` ise görüntü kusurlu (`0`), diğer durumda sağlam (`1`) olarak sınıflandırılır. Ağırlıklar önceki dış-kat tahminleri incelenerek seçilmiş **keşifsel** hiperparametrelerdir; bağımsız veride optimize edildiği iddia edilmez.

### Bölgesel eşleme

512×512 görüntü, 64 adet 64×64 bölgeye bölünür. Bölgesel Ortak MIL'e, **aynı görüntünün aynı grid bölgesine** ait toplam 48 özellik bir arada verilir. MIL yalnızca görüntü etiketiyle eğitildiğinden kusurlu görüntünün bütün bölgeleri otomatik olarak kusurlu olarak etiketlenmez.

Sağlam referans atlası yalnızca eğitim katındaki sağlam görüntülerden oluşturulur; test görüntüleri referans bankasına girmez. Farklı çekimlerde aynı grid koordinatının fiziksel olarak aynı noktayı göstermesi garanti değildir; görüntüler arası geometrik registration yapılmamaktadır.

### Donanım hızlandırması

Sabit özellik çıkarımında `PyTorch CUDA (eşitlik kontrollü, multi-stream) → Cython/OpenMP → OpenCV/NumPy` yönlendirmesi vardır. Morfoloji ve geometri kodları OpenCV/CPU kullanır. GPU için CUDA uyumlu PyTorch ve NVIDIA sürücüsü gerekir. Uçtan uca GPU hız üstünlüğü bağımsız olarak ölçülmemiştir.

## Veri kümesi ve kaynak atfı

**Kaynak:** Kaggle üzerinde **Ravirajsinh Dabhi** tarafından yayımlanan [Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) veri kümesi.

Deneylerde veri kümesinin **veri artırımı uygulanmamış `casting_512x512` bölümü** kullanılmıştır. Bu bölüm **1.300 adet 512×512 gri-seviye görüntü** içerir: **781 `def_front` (kusurlu)** ve **519 `ok_front` (sağlam)**. Kaynak sayfadaki artırılmış 300×300 görüntü koleksiyonundan farklıdır. Deneylerde görüntü düzeyinde etiketler kullanılmış; doğrulanmış piksel düzeyinde kusur maskeleri kullanılmamıştır.

Görüntüler bu kod deposuyla **dağıtılmaz**; orijinal Kaggle sayfasından edinilip aşağıdaki [Veri dizinleri](#veri-dizinleri) yapısına yerleştirilmelidir. **Kodun lisansı veri kümesine kullanım hakkı vermez**. Görüntüleri kullanmadan veya yeniden dağıtmadan önce veri kümesi sahibinin yayımladığı lisans ve kullanım koşulları incelenmelidir.

## Keşifsel performans

Yukarıda kaynak gösterilen Kaggle veri kümesinin **1.300** gri-seviye görüntüsü (**781 kusurlu, 519 sağlam**) kullanıldı. Üç farklı rastgele başlangıçla tekrarlanan 5-fold Stratified CV sonucu 3.900 test tahmini oluşturulmuştur; bunlar 1.300 bağımsız görüntünün üç kez değerlendirilmesidir.

CV Atlas MIL uygulaması, **400 ağaç ve 35 epoch** ayarlarıyla üç tekrarlı 5-fold protokolünün **15 dış katının tamamında yeniden eğitilerek** test edilmiştir.

| Yöntem ve değerlendirme kaynağı | Toplam hata | Kaçırılan kusur | Yanlış alarm | Ortalama doğruluk |
|---|---:|---:|---:|---:|
| Önceki Atlas-MIL (geçmiş CV) | 12 | 10 | 2 | %99,69 |
| Önceki Atlas-MIL %70 + morfoloji %30 (arşiv kayıtlarından) | 10 | 10 | 0 | %99,74 |
| **CV Atlas MIL — yeniden çalıştırılmış CV** | **9** | **8** | **1** | **%99,77** |
| Sabit füzyon ağırlıklarıyla önceki uzmanların skorlarının birleşimi | 8 | 8 | 0 | %99,79 |

**CV Atlas MIL uygulamasının yeniden eğitilip ölçülmüş sonucu 9 hatadır.** Önceki ayrı uzmanların arşiv tahminleriyle hesaplanan 8 hatalık sonuç yeni uygulamanın ölçümü değildir. Üç CV tekrarının sonuçları sırasıyla **3, 5 ve 1 hata** vermiştir.

Ağırlıklar, aynı veri üzerinde incelenen çok sayıda füzyon arasından sonradan seçildiği için **%99,77 bağımsız ve yansız genelleme tahmini değildir**. Mimari geliştirmede de aynı veriler kullanılmıştır; üretim koşullarında doğrulama gereklidir.
### Konumsal eşleştirme kontrolü

Yalnızca Bölgesel Ortak MIL üzerinde, tüm özellikler birlikte yer değiştirdiğinde toplam hata **51 → 54** oldu. Morfoloji ve atlas yoğunluk ailesinden ayrı kaydırıldığında **98**; ayrı ayrı karıştırıldıklarında iki tekrarda **146** ve **156** hata gözlendi. Bu, aynı bölgedeki özellik birlikteliğinin bölgesel MIL için önemli olduğunu destekler. Ancak beş uzmanlı nihai füzyonda bu etki ayrıca kanıtlanmış değildir.

### Çekim koşulu değişimi için yaklaşık test

Görüntü arka planının parlaklık özelliklerinden oluşturulan beş grupta, her grup sırayla testte bırakılmıştır. Bu gerçek kamera kimliği veya üretim partisi bazlı ayrım değildir.

| Yöntem | Toplam hata / 1.300 | Kaçırılan kusur | Yanlış alarm |
|---|---:|---:|---:|
| Önceki Atlas-MIL | 297 | 45 | 252 |
| **CV Atlas MIL — yeniden eğitim** | **271** | **20** | **251** |
| Önceki uzmanların skorlarıyla sabit ağırlıklı füzyon | 263 | 21 | 242 |
| Sadece yüksek çözünürlüklü morfoloji | **233** | 43 | 190 |

CV Atlas MIL uygulamasının çekim koşulu testinde doğruluğu **%79,15** olmuştur. Eski 263 hatalık sonuç, ayrı uzmanların daha önceki eğitimlerine dayanan arşiv skorlarından hesaplanmıştır. Dağılım değişimindeki belirgin performans düşüşü, çekim koşullarının ve arka planın hâlâ bozucu faktörler olduğunu gösterir. Rastgele CV doğruluğu tek başına üretimde genelleme garantisi oluşturmaz.

## Kurulum

Python 3.10+ tavsiye edilir.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

İsteğe bağlı Cython/OpenMP derlemesi:

```bash
python -m pip install -r requirements-build.txt
python setup.py build_ext --inplace
```

CUDA desteği için uygun PyTorch CUDA paketinin [resmî PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/) kurulması gerekir.

## Veri dizinleri

```text
DataSet/casting/
├── def_front/
│   ├── casting_def_001.jpeg
│   └── ...
└── ok_front/
    ├── casting_ok_001.jpeg
    └── ...
```

Görüntüler **512×512** çözünürlükte olmalıdır. JPEG, PNG, BMP ve TIFF desteklenir; varsayılan akışta yeniden boyutlandırma yapılmaz. Ham görüntüler ve eğitimden türetilen önbellekler kaynak kodu paketine dahil değildir.

## Modeli eğitme

```bash
python main.py train \
  --data-dir DataSet/casting \
  --backend auto \
  --model models/cv_atlas_mil.joblib
```

## Yeni görüntüleri değerlendirme

```bash
python main.py predict \
  --model models/cv_atlas_mil.joblib \
  --image NewImages \
  --backend auto \
  --output predictions.csv \
  --heatmap-dir heatmaps
```

`NewImages` klasöründeki görüntüler etiket gerektirmez. CSV dosyasında nihai kusur skoru ve beş uzmanın skorları bulunur. Isı haritaları kusur maskesi veya nedensel açıklama olarak yorumlanmamalıdır. `joblib`/pickle dosyaları yalnızca güvenilir kaynaklardan yüklenmelidir.

## Çapraz doğrulama

```bash
python main.py evaluate \
  --data-dir DataSet/casting \
  --seeds 20261025 20261026 20261027 \
  --folds 5 \
  --output evaluation
```

Kısa yazılım kontrolü için `--trees 12 --mil-epochs 1 --max-folds 1` kullanılabilir. Bu kontrol gerçek model performansının ölçümü yerine geçmez.

## Sınırlamalar

- Gerçek kusur konumlarının piksel düzeyinde doğrulanmış etiketleri yoktur.
- Aynı grid konumu, farklı bakış açılarında aynı fiziksel yüzey anlamına gelmeyebilir.
- Veri kümesinde arka plan ve çekim koşulları sınıf etiketiyle ilişkili olabilir.
- Ağırlıklar ve mimari, aynı veri kümesindeki çok sayıda deney üzerinden seçilmiştir.
- Gerçek CUDA performansı ve yeni üretim koşullarında doğruluk bağımsız olarak kanıtlanmamıştır.

## Lisans

Kaynak kodu **PolyForm Noncommercial License 1.0.0** kapsamında yayımlanır. [LICENSE.md](LICENSE.md). Ticari kullanım Zermind'den ayrı bir lisans gerektirir. Bu lisans OSI tanımına göre açık kaynak değil, **source-available** modelidir. Veri kümeleri ve bağımsız dağıtılan model ağırlıkları farklı kullanım haklarına tabi olabilir.
