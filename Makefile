.PHONY: help venv install install-sys setup sign verify lint clean clean-venv

VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# Defaults (override on command line)
NAME      ?=
OUTPUT    ?= ./signatures
SIG       ?=
THRESHOLD ?= 0.70
CHUNK     ?= 3.0
AUDIO     ?=

help:
	@echo "Utilizare:"
	@echo "  make setup                    Instalare completă (sistem + Python)"
	@echo "  make install-sys              Instalare pachete sistem (ffmpeg, libsndfile1)"
	@echo "  make install                  Instalare pachete Python în .venv"
	@echo "  make venv                     Creare mediu virtual .venv"
	@echo ""
	@echo "  make sign NAME='...' AUDIO='fisier1.mp3 fisier2.wav'"
	@echo "                                Creare semnătură vocală"
	@echo "  make verify SIG=signatures/nume.npy AUDIO='test.wav'"
	@echo "                                Verificare audio față de semnătură"
	@echo ""
	@echo "Opțiuni opționale:"
	@echo "  OUTPUT=./signatures           Director output pentru semnături (default: ./signatures)"
	@echo "  THRESHOLD=0.70                Prag cosine similarity (default: 0.70)"
	@echo "  CHUNK=3.0                     Lungime chunk în secunde (default: 3.0)"
	@echo ""
	@echo "  make lint                     Verificare sintaxă Python"
	@echo "  make clean                    Șterge directorul de semnături"
	@echo "  make clean-venv               Șterge mediul virtual .venv"

# ----------------------------------------------------------------------------
# Virtual environment
# ----------------------------------------------------------------------------

$(VENV)/bin/pip:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(VENV)/bin/pip

# ----------------------------------------------------------------------------
# Instalare
# ----------------------------------------------------------------------------

install-sys:
	sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg

install: venv
	$(PIP) install Cython packaging
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -r requirements.txt

setup: install-sys install

# ----------------------------------------------------------------------------
# Sign & Verify
# ----------------------------------------------------------------------------

sign: venv
	@test -n "$(NAME)"  || (echo "ERROR: NAME='Nume Vorbitor' este obligatoriu" && exit 1)
	@test -n "$(AUDIO)" || (echo "ERROR: AUDIO='fisier1.mp3 fisier2.wav' este obligatoriu" && exit 1)
	$(PYTHON) sign.py --name "$(NAME)" --output $(OUTPUT) --chunk $(CHUNK) $(AUDIO)

verify: venv
	@test -n "$(SIG)"   || (echo "ERROR: SIG=signatures/nume.npy este obligatoriu" && exit 1)
	@test -n "$(AUDIO)" || (echo "ERROR: AUDIO='test.wav' este obligatoriu" && exit 1)
	$(PYTHON) verify.py --signature $(SIG) --threshold $(THRESHOLD) --chunk $(CHUNK) $(AUDIO)

# ----------------------------------------------------------------------------
# Mentenanță
# ----------------------------------------------------------------------------

lint: venv
	$(PYTHON) -m py_compile sign.py   && echo "sign.py: OK"
	$(PYTHON) -m py_compile verify.py && echo "verify.py: OK"

clean:
	rm -rf $(OUTPUT)
	@echo "Removed $(OUTPUT)"

clean-venv:
	rm -rf $(VENV)
	@echo "Removed $(VENV)"
