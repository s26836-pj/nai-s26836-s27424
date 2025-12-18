# Plant AI API (Vision + Sensor) — PC Setup

Backend (FastAPI) do predykcji stanu rośliny na podstawie:
- **Vision AI**: klasyfikacja obrazu (TFLite)
- **Sensor AI**: multilabel z danych sensorów (TFLite + scaler) + reguły `too_bright_now`
- **Combined**: łączenie wyników w jedną odpowiedź

> Repo zawiera kod. Dataset (images + CSV) jest na Google Drive.

---

## 1) Wymagania

- Python **3.11+** (zalecane 3.11 lub 3.12)
- Windows / Linux / macOS
- (Opcjonalnie) InfluxDB v1 jeśli chcesz endpointy `*/live`

---

## 2) Struktura projektu (ważne)
> Uwaga: nazwy folderów mogą się nieznacznie różnić - kluczowe są katalogie

Zakładamy strukturę (przykład):
- images
- ai_api/
  - data/
    - influx_reader.py
  - sensor_model/
    - csv /
      - dane z csv
    - sensor_data/
      - zapisane modele/metryki po uruchumieniu skryptu
    - light_rules.py
    - sensor_inference.py
    - training_sensor_model.py
    - convert_to_tflite.py
  - vision_model/
  - main.py
  - vision_model/
    - convert_to_tflite.py
    - train_vision_model.py
    - vision_interface.py
  - vision_data/
    - zapisane modele/metryki po uruchumieniu skryptu
  - vision_inference.py
- build_plant_state.py
- main.py


Jeśli Twoje foldery różnią się nazwami dostosuj importy/ścieżki, ale idea jest ta sama:
- artefakty modeli trzymasz w `*/vision_data` i `*/sensor_data`.

---

## 3) Pobranie datasetu (Google Drive)

**Nie trzymamy** 400 zdjęć w repo. Pobierz dataset lokalnie i ustaw ścieżki.

1) Pobierz z Google Drive:
- Images: **https://drive.google.com/drive/folders/1NPhPGqxeY3_6QfzEsXZtQ2VtA8iZJ5ch?usp=sharing**

2) Rozpakuj do lokalnego folderu, zgodnie ze struktura projektu.

---

## 4) Konfiguracja środowiska (PC)

### Windows (PowerShell)
```powershell
cd <FOLDER_Z_REPO>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

---
## 5) Trenowanie modeli

Uruchom bedąc w odpowiedniej ścieżce do pliku:

`cd ai_api/vision_model`

`python ai_api/vision_model/train_vision_model.py`

`python ai_api/vision_model/convert_to_tflite.py`

`cd ai_api/sensor_model`

`python ai_api/sensor_model/training_sensor_model.py`

`python ai_api/sensor_model/convert_to_tflite.py`

---

## 6) Uruchomienie API (na PC)

### Start serwera:
`uvicorn ai_api.main:app --host 0.0.0.0 --port 8000 --reload`

---

## 7) Frontend (Plant UI)

Projekt posiada prosty frontend (`plant_ui`) do testowania API (upload zdjęcia, podgląd wyników).

### Wymagania
- Node.js **18+**
- npm

### Uruchomienie UI

W osobnym terminalu:

```bash
cd plant_ui
npm.cmd install
npm.cmd run dev
```

### Po uruchomieniu:

- UI będzie dostępne pod adresem:
- `http://localhost:5173` lub inny port podany w konsoli

> Uwaga

- Backend musi być uruchomiony (uvicorn ai_api.main:app ...)
- UI komunikuje się z API pod:
- `http://localhost:8000`
- Jeśli backend działa na innym adresie/porcie:

  - zmień URL w pliku konfiguracyjnym frontendu (np. .env lub api.ts)

### Kolejność uruchamiania na PC
1. Uruchom skrypty do treningu modeli
2. Uruchom backend (FastAPI)
3. Uruchom frontend (`plant_ui`)
4. Wejdź w przeglądarce na UI i testuj API