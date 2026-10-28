# Hoge Beek - neerslag en afvoer

Een Streamlit-app om neerslag om te rekenen naar afvoer en waterdiepte.
Je uploadt gebiedsbestanden, kiest een bui en downloadt de resultaten.

## Lokaal starten (Windows)

Installeer **Python 3.13** als je die nog niet hebt. Zorg dat `python` beschikbaar is in je terminal.
Open PowerShell in deze map (de map met `app.py`). Controleer met `python --version` of je Python 3.13 gebruikt.
Werk bij voorkeur vanuit een lokale kopie op je C:-schijf: een Python-omgeving op een netwerkschijf kan erg traag zijn.

Voer deze opdrachten eenmalig uit:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.venv` is een aparte Python-omgeving voor dit project. Activeren is niet nodig: we gebruiken hieronder direct de Python uit die map.

Start de app met:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 als de browser niet vanzelf opent. Stoppen: **Ctrl+C** in PowerShell.
De volgende keer hoef je alleen de startopdracht uit te voeren.

Op macOS/Linux: maak de omgeving met `python3.13 -m venv .venv` en gebruik `.venv/bin/python` in plaats van `.\.venv\Scripts\python.exe` bij de overige opdrachten.

## Gebruiken

1. Upload geodata (`.parquet`), catchments (`.gpkg`) en inlets (`.gpkg`). Deze gebiedsbestanden zijn niet meegeleverd.
2. Kies **Waterinfo** en haal neerslag op, of kies **Eigen CSV** en upload een bui. Waterinfo heeft internet nodig.
3. Klik op **Berekenen**.
4. Bekijk de grafieken en download de resultaat-ZIP voordat je de sessie sluit.

Voor een eigen bui kun je [dit voorbeeld](examples/neerslag_15min.csv) gebruiken: 6 mm in een uur, met intervallen van 15 minuten.
De CSV-kolommen zijn `timestamp,precipitation_mm`: tijdstip met tijdzone-offset en mm per interval. Neem ook droge intervallen op met waarde 0.

Eigen gebiedsbestanden maken? Zie [de handleiding](input_voorbereiding/Input_voorbereiding.pdf) en [het voorbereidingsscript](input_voorbereiding/prepare_model_inputs.py).
Installeer hiervoor ook `whitebox` in dezelfde Python-omgeving:

```powershell
.\.venv\Scripts\python.exe -m pip install whitebox
.\.venv\Scripts\python.exe input_voorbereiding/prepare_model_inputs.py --help
```

Whitebox downloadt bij het eerste gebruik zijn rekenprogramma; daarvoor is internet nodig.

## Naar GitHub en Streamlit Cloud

Installeer [Git](https://git-scm.com/downloads) en maak op GitHub een **lege repository** (zonder README, .gitignore of licentie).
Voer in deze projectmap uit; vervang de naam, het e-mailadres en de GitHub-URL door je eigen gegevens:

```powershell
git init -b main
git config user.name "Jouw naam"
git config user.email "jij@example.com"
git add .
git status
git commit -m "Eerste versie"
git remote add origin https://github.com/JOUW-NAAM/JOUW-REPO.git
git push -u origin main
```

`.gitignore` houdt de Python-omgeving, tijdelijke bestanden en lokale secrets uit Git.
Bekijk bij `git status` welke bestanden je gaat uploaden. Gebiedsbestanden kun je buiten deze map bewaren en later in de app uploaden.

Ga naar [Streamlit Community Cloud](https://share.streamlit.io/) en verbind je GitHub-account.
Klik op **Create app**, kies je repository, branch **main** en bestand **app.py**.
Kies bij **Advanced settings** Python **3.13** en klik op **Deploy**.
Streamlit installeert de pakketten uit `requirements.txt`; je hoeft `.venv` niet te uploaden.
Zie ook de [Streamlit-deployhandleiding](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Latere wijzigingen publiceren:

```powershell
git add .
git commit -m "Beschrijf je wijziging"
git push
```
