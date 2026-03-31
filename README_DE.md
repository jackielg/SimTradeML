# SimTradeML

[English](./README.md) | [中文](./README_CN.md) | Deutsch

**PTrade-kompatibles quantitatives ML-Framework** — hilft Benutzern, Vorhersagemodelle schnell zu trainieren, die in SimTradeLab und PTrade verwendet werden können.

## Kernpositionierung

SimTradeML ist die **Machine-Learning-Toolchain** für [SimTradeLab](https://github.com/kay-ou/SimTradeLab):
- 🎯 **Für PTrade optimiert**: Trainierte Modelle können direkt im SimTradeLab-Backtesting und PTrade-Live-Handel verwendet werden
- ⚡ **Schnelles Training**: Von Daten zum einsatzfähigen Modell in 5 Minuten
- 📊 **Quantitative Finanzkennzahlen**: Professionelle Bewertung mit IC/ICIR/Quantilrenditen
- 🔧 **A-Aktien-Ökosystem-Integration**: Tiefe Integration mit SimTradeLab-Datenquellen

## Schnellstart

### Installation

```bash
cd /path/to/SimTradeML
poetry install
pip install simtradelab  # Falls SimTradeLab-Datenquelle benötigt wird
```

### Erstes Modell in 5 Minuten trainieren

```bash
# 1. Daten vorbereiten (SimTradeLab h5-Dateien ins data/-Verzeichnis kopieren)
mkdir -p data
cp /path/to/ptrade_data.h5 data/
cp /path/to/ptrade_fundamentals.h5 data/

# 2. Vollständige Trainingspipeline ausführen
poetry run python examples/mvp_train.py
```

### Vollständige Beispiele

Siehe `examples/`-Verzeichnis:
- **mvp_train.py** — Vollständige Trainingspipeline (Datensammlung, Training, Export)
- **complete_example.py** — Empfohlene Verwendung (Einzeldatei-Paket)

### Empfohlene Verwendung (Einzeldatei-Paket)

```python
from simtrademl.core.models import PTradeModelPackage

# Nach dem Training speichern (eine Datei enthält alles)
package = PTradeModelPackage(model=model, scaler=scaler, metadata=metadata)
package.save('my_model.ptp')

# In PTrade laden und vorhersagen
package = PTradeModelPackage.load('my_model.ptp')
prediction = package.predict(features_dict)  # Auto-Validierung + Skalierung
```

## Kernfunktionen

### PTrade-Kompatibilität
- ✅ **XGBoost 1.7.4**: an die aktuelle PTrade-Umgebung angepasst
- ✅ **Flexible Speicherformate**: JSON, Pickle, XGBoost-nativ
- ✅ **Plug and Play**: Trainierte Modelle direkt in SimTradeLab verwendbar

### ML-Fähigkeiten
- **Datenquellenabstraktion**: Einfacher Wechsel zwischen verschiedenen Datenquellen
- **Feature-Engineering**: Eingebaute technische Indikatoren, unterstützt eigene Features
- **Bewertungsmetriken**: IC/ICIR/Quantilrenditen/Richtungsgenauigkeit
- **Parallelverarbeitung**: Automatische Multi-Prozess-Beschleunigung

### Quantitative Finanzspezialisierung
- **Zeitreihen-Strenge**: Verhindert Future Data Leakage
- **Tägliches Rebalancing**: Simuliert reale Handelsszenarien
- **Quantilrenditen**: Strategie-Renditesimulation
- **Richtungsgenauigkeit**: Bewertung der Auf-/Abwärtsprognose

## Projektstruktur

```
src/simtrademl/
├── core/
│   ├── data/          # Datenschicht (DataSource, DataCollector)
│   └── utils/         # Hilfsfunktionen (Config, Logger, Metrics)
└── data_sources/      # Datenquellenimplementierungen
    └── simtradelab_source.py

examples/
└── mvp_train.py       # Vollständiges Trainingsbeispiel
```

## Konfigurationsbeispiel

```yaml
data:
  lookback_days: 60
  predict_days: 5
  sampling_window_days: 15

model:
  type: xgboost
  params:
    max_depth: 4
    learning_rate: 0.04
    subsample: 0.7
    colsample_bytree: 0.7

training:
  train_ratio: 0.70
  val_ratio: 0.15
  parallel_jobs: -1  # -1 = alle CPUs nutzen
```

## Tests

```bash
poetry run pytest
poetry run pytest --cov=simtrademl --cov-report=html
```

## Abhängigkeiten

**Kern**: Python 3.9+, numpy, pandas, scikit-learn, **xgboost 1.7.4** (an die aktuelle PTrade-Umgebung angepasst)
**Optional**: simtradelab (Daten), optuna (Hyperparameteroptimierung), mlflow (Experiment-Tracking)

> ⚠️ **Wichtig**: Die reale PTrade-Umgebung verwendet derzeit XGBoost 1.7.4, und dieses Projekt ist jetzt darauf abgestimmt.

## Lizenz

MIT-Lizenz

---

**Dokumentation**: Siehe `examples/mvp_train.py` für vollständige Beispiele
**Issues**: Auf GitHub einreichen
**Testabdeckung**: 88% | Alle 66 Tests bestanden
