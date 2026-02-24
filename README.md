# Football Action Recognition — ML Project

**Tim:** Stefan Paunović (RA 27/2022) i Tamás Papp (RA 4/2022)
**Asistent:** Teodor Vidaković

Sistem za automatsko prepoznavanje i klasifikaciju sportskih akcija u fudbalskim utakmicama. Model klasifikuje slike u 6 kategorija: **gol, šut na gol, slobodnjak, ofsajd, karton, normalna igra**.

---

## Instalacija

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/Pappt04/Football-Action-Recognizer-ML.git
cd Football-Action-Recognizer-ML
```

### 2. Virtuelno okruženje

```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
python -m pip install -r requirements.txt
```

### 3. Preuzimanje dataseta

Dataset nije uključen u repozitorijum (1.7 GB). Preuzeti sa Kaggle:

**Soccer View and Event Score** — https://www.kaggle.com/datasets/fahadahmedkhokhar/soccer-view-and-event-score

Nakon preuzimanja, raspakovati arhivu tako da struktura foldera izgleda ovako:

```
data/
└── soccer_events/
    └── Dataset/
        └── EventClasses/
            ├── goal/
            ├── gattempts/
            ├── freekick/
            ├── corner/
            ├── offside/
            ├── yellowcard/
            ├── redc/
            ├── playercelebration/
            ├── plentystock/
            └── spectator/
```

---

## Pokretanje

```bash
source venv/bin/activate

python main.py            # puno treniranje (~37k slika, 5-7h na CPU)
python main.py --sample 50  # 50% podataka (~1.5-2h)
python main.py --quick    # 5% podataka, 3 epohe (~5 min, samo test)
```

---

## Arhitektura modela

```
Input (224x224x3)
    → EfficientNetB0 (pretreniran na ImageNet, frozen u fazi 1)
    → GlobalAveragePooling2D
    → Dense(256, ReLU)
    → Dropout(0.5)
    → Dense(6, Softmax)
```

**Treniranje u dve faze:**
- Faza 1: frozen backbone, lr=1e-4, 10 epoha
- Faza 2: odmrznut block5+, lr=1e-5, 20 epoha

**Mapiranje klasa (10 → 6):**

| Originalna klasa | Finalna klasa |
|-----------------|---------------|
| goal | goal |
| gattempts | goal_attempt |
| freekick + corner | free_kick |
| offside | offside |
| yellowcard + redc | card |
| playercelebration + plentystock | normal_play |
| spectator | isključeno |

---

## Rezultati

Sve metrike i vizualizacije se čuvaju u `outputs/` nakon treniranja:

| Fajl | Sadržaj |
|------|---------|
| `best_model.keras` | Sačuvani model sa najboljim val_loss |
| `training_history.png` | Loss i accuracy kroz epohe (faza 1 + 2) |
| `confusion_matrix.png` | Matrica konfuzije (normalizovana + apsolutna) |
| `gradcam_examples.png` | Grad-CAM vizualizacija po klasama |
| `false_positives.png` | Primeri pogrešnih predikcija sa confidence |
| `per_class_f1.png` | F1-score po klasama |
| `baseline_comparison.png` | Poređenje sa majority/random baseline |
| `classification_report.txt` | Precision, recall, F1 po klasi |
| `results_summary.json` | Sve metrike u JSON formatu |

---

## Ciljne metrike

- Top-1 Accuracy > 75%
- Weighted F1-score > 0.72
