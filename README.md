# Heart Failure - code

Eén Git-repository voor alle opdrachten van de Heart Failure-case.

- `opdracht1/procesmodel.py`: uitvoerbaar procesmodel van het huidige behandelproces.
- `Heart_failure_start.ipynb`: bestaand startnotebook voor latere analyse en ML-opdrachten.
- Voeg code voor volgende opdrachten toe in aparte mappen wanneer die nodig is. Gedeelde functies kunnen later in `src/` komen.

Uitvoeren vanuit deze map:

```powershell
python opdracht1/procesmodel.py
```

Het script vraagt in de terminal om instroom (`huisarts`/`seh`), of sprake is van een hartinfarct (`ja`/`nee`), de bekende vervolgroute (`ic`/`regulier`/`vertrek`) of uitkomst (`overlijden`), en operatie (`ja`/`nee`). Bij een operatie volgt een vrij tekstveld voor de uitgevoerde ingreep; de omschrijving verschijnt in de uitvoer. Daarna toont het de procesroute met onderbouwing. Per uitvoering kun je maximaal één scenario invoeren; na het tonen van de route sluit het script af. Ongeldige antwoorden worden opnieuw gevraagd; Ctrl+C sluit af. Invoer wordt niet opgeslagen.

Deze keuzes zijn handmatige scenario-invoer. Het model berekent geen diagnose of behandelkeuze uit medische meetwaarden. Het script gebruikt de actuele stapcodes uit het notebook, inclusief E01 (huisarts), E02 (SEH) en E03 (overlijden). Toekomstige wijzigingen worden niet automatisch tussen script en notebook gesynchroniseerd.

Bronmateriaal en uitgebreide uitleg staan buiten deze repository in `../informatie/`. De dataset staat in `../data/`. Open het startnotebook met deze codemap als werkmap; het leest de dataset via `../data/Heart faillure prediction data.csv`.

Deze README bevat alleen technische startinstructies. De inhoudelijke uitleg van opdracht 1 staat in `../informatie/opdracht1/README.md`.

Er is geen remote ingesteld. Controleer notebooks op opgeslagen persoonsgegevens voordat je ze commit of publiceert.

Modelaanname in het terminalscript: `hartinfarct=ja` vertegenwoordigt de directe IC-route. De case noemt daarvoor een ernstig geval met evident hartinfarct. De invoer is een bekende beoordeling, geen diagnose door het script. De omschrijving van de ingreep wordt alleen weergegeven. Het notebook gebruikt nog de eerdere parameter `ernstig`.
