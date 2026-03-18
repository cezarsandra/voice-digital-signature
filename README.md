# Voice Digital Signature

Creează și verifică semnătura digitală vocală a unui vorbitor din fișiere audio (mp3/wav), folosind modelul NeMo **TitaNet-Large**.

---

## Cum funcționează

1. Fiecare fișier audio este convertit la 16kHz mono WAV (automat via ffmpeg)
2. Audio-ul este segmentat în chunk-uri de ~3 secunde
3. TitaNet-Large extrage un vector de embedding (192D) per chunk
4. Toate embedding-urile sunt agregate prin medie și L2-normalizate
5. Rezultatul este semnătura vocală — un fișier `.npy` compatibil cu proiectul `nemo-diarization`

Cu cât mai multe fișiere și mai mult audio, cu atât semnătura este mai robustă.

---

## Instalare

### Cerințe sistem
- Python 3.10+
- ffmpeg instalat (`sudo apt install ffmpeg`)
- CUDA opțional (merge și pe CPU, mai lent)

### Creare mediu virtual și instalare dependențe

```bash
make install
```

sau manual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install Cython packaging
pip install -r requirements.txt
```

---

## Utilizare

### 1. Creare semnătură (enrollment)

```bash
python sign.py --name "Nume Vorbitor" fisier1.mp3 fisier2.wav fisier3.mp3
```

**Opțiuni:**

| Opțiune | Prescurtare | Default | Descriere |
|---|---|---|---|
| `--name` | `-n` | *obligatoriu* | Numele vorbitorului |
| `--output` | `-o` | `.` | Director unde se salvează semnătura |
| `--chunk` | `-c` | `3.0` | Lungimea chunk-urilor în secunde |
| `--device` | | auto | `cuda` sau `cpu` |

**Output:**
- `<nume>.npy` — semnătura vocală (vector float32, 192D)
- `<nume>_meta.json` — metadata: număr fișiere, chunk-uri, durată totală

**Exemplu:**
```bash
python sign.py --name "Ion Popescu" --output ./signatures/ rec1.mp3 rec2.wav
```

---

### 2. Verificare (verification)

```bash
python verify.py --signature signatures/ion_popescu.npy test.wav
```

**Opțiuni:**

| Opțiune | Prescurtare | Default | Descriere |
|---|---|---|---|
| `--signature` | `-s` | *obligatoriu* | Calea către fișierul `.npy` |
| `--threshold` | `-t` | `0.70` | Prag cosine similarity pentru MATCH |
| `--chunk` | `-c` | `3.0` | Lungimea chunk-urilor în secunde |
| `--device` | | auto | `cuda` sau `cpu` |

**Output:**
- Similarity medie, maximă, minimă per fișier
- Verdict: `MATCH` sau `NO MATCH`
- Exit code `0` dacă toate fișierele se potrivesc, `1` altfel

**Verificare batch:**
```bash
python verify.py --signature signatures/ion_popescu.npy *.wav --threshold 0.75
```

---

## Ajustarea pragului (threshold)

| Threshold | Comportament |
|---|---|
| `0.60` | Permisiv — acceptă variații mari de calitate audio |
| `0.70` | **Default** — echilibru bun între securitate și toleranță |
| `0.80` | Strict — necesită audio de calitate similară cu enrollment |
| `0.85+` | Foarte strict — aproape identic cu înregistrările de referință |

---

## Compatibilitate cu nemo-diarization

Fișierele `.npy` generate sunt compatibile direct cu funcția `identify_speakers()` din proiectul `nemo-diarization`:

```python
import numpy as np

signatures = {
    "Ion Popescu": np.load("signatures/ion_popescu.npy"),
    "Maria Ionescu": np.load("signatures/maria_ionescu.npy"),
}
# Pasează direct în identify_speakers(segments, audio_path, signatures, threshold, ...)
```

---

## Structura proiectului

```
voice-digital-signature/
├── sign.py          # Creare semnătură din fișiere audio
├── verify.py        # Verificare audio față de semnătură
├── requirements.txt
├── Makefile
└── README.md
```

---

## Recomandări pentru o semnătură puternică

- Folosește **minim 3-5 fișiere** audio diferite (zile diferite, microfoane diferite)
- Durata totală recomandată: **cel puțin 2-3 minute** de vorbire curată
- Evită zgomot de fond puternic în fișierele de enrollment
- Dacă verificarea dă false negative, coboară threshold-ul sau adaugă mai multe fișiere la enrollment
