# Rozpoznávanie ľudských činností z videa

Materiály a návody k predmetu **Hlboké neurónové siete** (FEI STU Bratislava).

Notebooky ukazujú, ako v prostredí Python (PyTorch, scikit-learn) rozpoznať z krátkeho videoklipu,
ktorú činnosť človek vykonáva. Porovnávajú sa modely CNN-SVM, CNN-LSTM a video CNN
a najlepší sa aplikuje na kameru.

Aktuálne triedy: **`eat`** (jedenie) a **`clap`** (tlieskanie). Triedy sa nastavujú
v `har-hmdb51/har/config.py` (`CLASSES`); zvyšok kódu ich preberá odtiaľ.

| Model | Príznaky | Ako sa spracuje čas |
|---|---|---|
| CNN-SVM | ResNet18 (ImageNet), 512 na snímku | priemer cez 16 snímok |
| CNN-LSTM | ResNet18 (ImageNet), 512 na snímku | LSTM číta snímky postupne |
| R(2+1)D + SVM | video sieť R(2+1)D-18 (Kinetics-400), 512 na klip | 3D konvolúcie v sieti |
| R(2+1)D + LSTM | R(2+1)D-18, 4 úseky po 8 snímok  (4, 512) na klip | 3D konvolúcie + LSTM cez úseky |
| R(2+1)D doladená | tá istá sieť, posledný blok (`layer4`) dotrénovaný | 3D konvolúcie v sieti |

## Štruktúra

Všetky súbory sú v priečinku `har-hmdb51/`.

| Súbor | Obsah |
|---|---|
| `har/config.py` | nastavenia: triedy, cesty, počet snímok, počet delení |
| `har/data.py` | načítanie snímok a príprava vstupu pre siete |
| `har/models.py` | LSTM klasifikátor a jeho trénovanie |
| `har/evaluation.py` | delenie dát, hodnotenie modelov, metriky, grafy |
| `convert_hmdb_videos.py` | prevod videí HMDB51 (*.avi) na snímky |
| `01_data.ipynb` | zoznam klipov (`index.csv`), delenie podľa filmu, ukážky |
| `02_features_resnet.ipynb` | príznaky snímok pomocou ResNet18 → `features_resnet.npy` |
| `03_features_video.ipynb` | predpoveď R(2+1)D bez trénovania, príznaky klipov → `features_r21d.npy`, `features_r21d_seq.npy` |
| `04_cnn_svm.ipynb` | model CNN-SVM |
| `05_cnn_lstm.ipynb` | model CNN-LSTM |
| `06_video_cnn.ipynb` | model R(2+1)D + SVM, uloženie finálneho modelu pre kameru |
| `06b_video_finetune.ipynb` | doladenie (fine-tuning) siete R(2+1)D |
| `06c_video_lstm.ipynb` | model R(2+1)D + LSTM |
| `07_comparison.ipynb` | porovnanie všetkých modelov |
| `08_camera.py` | rozpoznávanie z kamery alebo videa v reálnom čase |

Notebooky 04–06c majú rovnakú štruktúru: popis → načítanie dát → hodnotenie →
(priebeh trénovania) → chyby modelu. Predikcie na testovacích dátach sa ukladajú do `results/`.

## Dáta

**HMDB51** – pôvodné videá (klipy z filmov a YouTube, 30 snímok/s) z oficiálneho repozitára
https://huggingface.co/datasets/Serrelab/hmdb51 (nie sú súčasťou repozitára).
Príkazy sa spúšťajú v priečinku `har-hmdb51/`, pre každú triedu z `CLASSES`:

```bash
sudo apt install unrar
wget -c "https://huggingface.co/datasets/Serrelab/hmdb51/resolve/main/hmdb51_org.rar"
unrar x hmdb51_org.rar eat.rar clap.rar      # z veľkého archívu len potrebné triedy
unrar x eat.rar videos/                      # videos/eat/*.avi
unrar x clap.rar videos/                     #  videos/clap/*.avi
python convert_hmdb_videos.py                #  HMDB51/<trieda>/<klip>/<snímka>.jpg
```

Aktuálne dáta: 238 klipov (`clap` 130, `eat` 108) z 90 filmov.

## Inštalácia

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python scikit-learn pandas matplotlib joblib ipykernel
```

V editore vyberte ako kernel notebookov `venv`. Odporúča sa GPU s CUDA, ale všetko beží aj na CPU.
Notebooky sa spúšťajú z priečinka `har-hmdb51/` (kvôli importu `har`).
Po zmene súborov v `har/` treba reštartovať kernel notebookov.

## Postup

Notebooky spúšťajte v poradí `01` → `07`. Potom:

```bash
python 08_camera.py                      # webkamera
python 08_camera.py --source video.mp4   # videosúbor
python 08_camera.py --source video.mp4 --out vystup.mp4 --no-window
```

## Protokol hodnotenia

- Delenie train/test (70/30) **podľa filmu** – klipy z jedného filmu nie sú naraz
  v trénovacích aj testovacích dátach.
- Každý model sa hodnotí na **10 rôznych deleniach**, výsledok je priemer ± smerodajná odchýlka.
- Parametre sa vyberajú len na trénovacích dátach (krížová validácia pre SVM,
  validačná časť a early stopping pre LSTM a doladenie).
- Metriky na testovacích dátach:
  - **presnosť** (*accuracy*) – koľko % testovacích klipov model zaradil správne,
  - **správne &lt;trieda&gt;** (*recall*) – koľko % klipov danej triedy model zaradil správne.
- Matica zámen (*confusion matrix*) sa zobrazuje v počtoch klipov aj v percentách.
- Referencia: model, ktorý vždy predpovedá najčastejšiu triedu v trénovacích dátach.

## Aktuálne parametre

| Časť | Parameter | Hodnota |
|---|---|---|
| Dáta | snímok na klip | 16, rovnomerne vybrané |
| | veľkosť snímky | 224 × 224 (R(2+1)D: zmenšenie a orezanie na 112 × 112) |
| | delenia train/test | 10 × (70 % / 30 %), podľa filmu |
| SVM | kernel a C | lineárny: C ∈ {0,001; 0,01; 0,1; 1}; RBF: C ∈ {0,1; 1; 10; 100}, γ ∈ {scale; 1e-3; 1e-4} |
| | výber parametrov | 5-násobná krížová validácia podľa filmu, metrika *accuracy* |
| | ďalšie | štandardizácia príznakov, `class_weight="balanced"` |
| | finálny model (kamera) | lineárny kernel, C = 0,01 |
| LSTM | architektúra | 1 vrstva, 128 neurónov, dropout 0,5 |
| | trénovanie | Adam, lr 1e-3, weight decay 1e-4, dávka 32, 40 epoch |
| | validácia | 20 % filmov z trénovacích dát, early stopping podľa validačnej straty |
| | vstup 06c | 32 snímok  4 úseky po 8 (4, 512) |
| Doladenie | trénované vrstvy | `layer4` + nová výstupná vrstva (dropout 0,5) |
| | trénovanie | Adam, lr 1e-5, weight decay 1e-4, dávka 4, 8 epoch, float16 na GPU |
| | ďalšie | zrkadlové otočenie (augmentácia), BatchNorm v režime inferencie, early stopping |
| Kamera | okno / interval predikcie | 3 s / 0,5 s |
| | prah neistoty / priemerovanie | \|skóre\| < 0,5 / posledné 3 predikcie |

## Výsledky

Priemer ± smerodajná odchýlka cez 10 delení, testovacie dáta.

| Model | presnosť (*accuracy*) | správne eat (*recall*) | správne clap (*recall*) |
|---|---|---|---|
| Najčastejšia trieda | 43 % ± 9 % | 40 % ± 52 % | 60 % ± 52 % |
| CNN-SVM | 87 % ± 7 % | 93 % ± 6 % | 82 % ± 11 % |
| CNN-LSTM | 86 % ± 8 % | 94 % ± 4 % | 79 % ± 16 % |
| R(2+1)D + SVM | 94 % ± 3 % | 98 % ± 4 % | 91 % ± 5 % |
| R(2+1)D + LSTM | 91 % ± 4 % | 99 % ± 3 % | 86 % ± 8 % |
| R(2+1)D doladená | – | – | – |

R(2+1)D bez trénovania (súčet pravdepodobností zodpovedajúcich tried Kinetics, notebook 3): 90 %.
Výsledok doladenej siete (`06b`) doplniť po jej spustení.

## Porovnanie s predchádzajúcimi verziami

Predchádzajúce verzie projektu rozpoznávali iné dvojice tried. Hlavnou metrikou bola
*balanced accuracy* (priemer *recall* oboch tried), náhodný odhad / väčšinová trieda = 0,50.
Pôvodné videá HMDB51, priemer cez 10 delení.

| Model | eat / drink (pôvodná hlavná úloha) | clap / wave | eat / clap |
|---|---|---|---|
| CNN-SVM (ResNet18) | 0,62 | 0,65 | 0,87 |
| CNN-LSTM (ResNet18) | 0,61 | 0,68 | 0,87 |
| R(2+1)D + SVM | **0,80 ± 0,02** | **0,78** | **0,94** |
| R(2+1)D doladená | 0,76 ± 0,04 | – | – |


Rozdiely medzi verziami:

| | predchádzajúca verzia | aktuálna verzia |
|---|---|---|
| triedy | `eat`, `drink` (skúšané aj `clap`, `wave`) | `eat`, `clap` |
| dáta | HMDB51 (+ voliteľne Charades) | HMDB51 |
| metriky | *accuracy*, *balanced accuracy*, *F1* | *accuracy*, *recall* pre každú triedu |
| kód | jeden súbor `har_utils.py` | balík `har/` (config, data, models, evaluation) |
| modely | CNN-SVM, CNN-LSTM, R(2+1)D + SVM, R(2+1)D doladená | + R(2+1)D + LSTM |
