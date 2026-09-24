"""Opdracht 1: voer een patientscenario in en toon het behandelproces.

Start vanuit deze map met: python procesmodel.py
De ingevoerde beoordelingen bepalen de route; dit script berekent geen diagnose.
De modelcode volgt het notebook. Beide bestanden synchroniseren niet automatisch.
"""

# Een dictionary koppelt elke unieke procescode aan een leesbare omschrijving.
STAPPEN = {
    # E01 en E02 zijn alternatieve startgebeurtenissen: de gekozen instroom bepaalt welke wordt gebruikt.
    'E01': 'Binnenkomst via huisarts',
    'E02': 'Binnenkomst via spoedeisende hulp',
    # S-codes beschrijven activiteiten of verblijf; ze kunnen in de routelijst voorkomen.
    'S01': 'Urgente opname: verpleegkundigen verzorgen eerste zorg en informatie',
    # D-codes zijn labels voor onderbouwing en worden niet als aparte routestap afgedrukt.
    'D01': 'Is er sprake van een hartinfarct?',
    'S02': 'Direct naar intensive care',
    'S03': 'Specialisten meten patiëntwaarden (meestal enkele uren)',
    'S04': 'Artsenteam beoordeelt gegevens en stelt diagnose op basis van ervaring en studies',
    'D02': 'Vervolgroute volgens gekozen scenario (criteria niet beschreven)',
    'S05': 'Verblijf op intensive care',
    'S06': 'Reguliere ziekenhuisopname',
    'S07': 'Ziekenhuis direct verlaten',
    # Operatie en overlijden worden apart bewaard: hun tijdstip is niet vastgelegd in de case.
    'S08': 'Operatie',
    'E03': 'Overlijden (tijdstip in het proces onbekend)',
}

# Deze koppeling wordt gedeeld door het model, de uitvoer en de invoervraag.
# None betekent dat geen verblijf-/vertrekroute bekend is bij de uitkomst overlijden.
VERVOLGSTAPPEN = {'ic': 'S05', 'regulier': 'S06', 'vertrek': 'S07', 'overlijden': None}


def modelleer_proces(instroom, hartinfarct, vervolg, operatie=False):
    """Geef de hoofdroute en losse onderdelen terug voor de ingevoerde keuzes."""
    # Controleer de invoer ook als de functie buiten de terminal wordt aangeroepen.
    if instroom not in ('huisarts', 'seh'):
        raise ValueError('Instroom moet huisarts of seh zijn.')
    if type(hartinfarct) is not bool or type(operatie) is not bool:
        raise ValueError('Hartinfarct en operatie moeten True of False zijn.')
    if vervolg not in VERVOLGSTAPPEN:
        raise ValueError('Onbekende vervolgroute.')

    # Kies de startgebeurtenis; eerste zorg vindt in beide routes plaats.
    start = 'E01' if instroom == 'huisarts' else 'E02'
    route = [start, 'S01']
    if hartinfarct:
        # Modelaanname: een ingevoerd hartinfarct vertegenwoordigt hier de directe IC-route.
        # De case beschrijft die route voor een ernstig geval met evident hartinfarct.
        route.append('S02')
    # extend() voegt meerdere stappen toe: onderzoek en diagnose.
    route.extend(['S03', 'S04'])
    vervolgstap = VERVOLGSTAPPEN[vervolg]
    if vervolgstap is not None:
        route.append(vervolgstap)

    # Operatie en overlijden blijven apart omdat hun timing niet is beschreven.
    gebeurtenissen = []
    if operatie:
        gebeurtenissen.append('S08')
    if vervolg == 'overlijden':
        gebeurtenissen.append('E03')
    return route, gebeurtenissen


def toon_scenario(naam, instroom, hartinfarct, vervolg, operatie=False, ingreep=""):
    """Print de route met beslissingen als onderbouwing onder de betreffende stap."""
    route, gebeurtenissen = modelleer_proces(instroom, hartinfarct, vervolg, operatie)

    # Bij een operatie is een omschrijving nodig; deze tekst bepaalt geen medische route.
    if operatie and (not isinstance(ingreep, str) or not ingreep.strip()):
        raise ValueError('Vul bij een operatie in welke ingreep heeft plaatsgevonden.')

    # Een dictionary koppelt een procescode aan de uitleg bij die stap.
    onderbouwing = {}
    if hartinfarct:
        onderbouwing['S02'] = (
            'D01: hartinfarct=ja ingevoerd; in dit model wordt de directe IC-route gevolgd.'
        )
    vervolgstap = VERVOLGSTAPPEN[vervolg]
    if vervolgstap is not None:
        onderbouwing[vervolgstap] = (
            f'D02: vervolg={vervolg!r} is vooraf gekozen in dit scenario. '
            'De case geeft hiervoor geen medische besliscriteria.'
        )
    if operatie:
        onderbouwing['S08'] = (
            'operatie=True is ingevoerd; de case geeft geen operatiecriteria.'
        )
    if vervolg == 'overlijden':
        onderbouwing['E03'] = (
            'vervolg=overlijden is ingevoerd; '
            'de case geeft geen criteria voor overlijdenstijdstip.'
        )

    # + combineert de lijsten voor de weergave; de losse onderdelen krijgen geen tijdstip.
    print(f'\n{naam} | instroom: {instroom} | hartinfarct: {"ja" if hartinfarct else "nee"}')
    for stap in route + gebeurtenissen:
        print(f'  {stap}: {STAPPEN[stap]}')
        if stap == 'S08':
            # Toon de letterlijk ingevulde omschrijving; beoordeel de inhoud niet.
            print(f'    Ingreep: {ingreep.strip()}')
        if stap in onderbouwing:
            print(f'    Onderbouwing - {onderbouwing[stap]}')
    if gebeurtenissen:
        print('Toelichting: de positie van operatie/overlijden in deze uitvoer legt geen tijdstip vast.')
    if vervolg == 'overlijden':
        print('De hoofdroute is illustratief; niet bekend is welke stappen aan het overlijden voorafgingen.')


def vraag_keuze(vraag, opties):
    """Vraag een keuze tussen vierkante haken en herhaal alleen bij ongeldige invoer."""
    keuzetekst = '/'.join(opties)
    while True:
        # input() leest tekst; strip() en lower() maken spaties en hoofdletters onbelangrijk.
        antwoord = input(f'{vraag} [{keuzetekst}]: ').strip().lower()
        if antwoord in opties:
            return antwoord
        print(f'Ongeldige invoer. Kies uit [{keuzetekst}].')


def main():
    """Vraag precies een scenario uit, toon het proces en sluit daarna af."""
    print('Heart Failure - behandelproces')
    print('Voer de bekende beoordeling en vervolgkeuze in; het model berekent deze niet.')
    print('De invoer wordt niet opgeslagen. Stoppen kan met Ctrl+C.\n')

    # De vergelijking met 'ja' zet het antwoord om naar True of False.
    instroom = vraag_keuze('Binnenkomst via', ('huisarts', 'seh'))
    hartinfarct = vraag_keuze('Is er sprake van een hartinfarct?', ('ja', 'nee')) == 'ja'
    vervolg = vraag_keuze('Vervolgroute of uitkomst', VERVOLGSTAPPEN)
    operatie = vraag_keuze('Heeft er een operatie plaatsgevonden?', ('ja', 'nee')) == 'ja'
    ingreep = ''
    if operatie:
        # Vrije tekst: behoud hoofdletters en verwijder alleen spaties aan de randen.
        ingreep = input('Welke ingreep heeft plaatsgevonden? [vrije tekst]: ').strip()
        while not ingreep:
            print('Vul een omschrijving van de ingreep in.')
            ingreep = input('Welke ingreep heeft plaatsgevonden? [vrije tekst]: ').strip()

    # Er is geen herhaallus: na deze uitvoer eindigt het programma.
    toon_scenario('Ingevoerd scenario', instroom, hartinfarct, vervolg, operatie, ingreep)


# Start de terminalvragen alleen bij rechtstreeks uitvoeren, niet bij importeren.
if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C of afgebroken invoer sluit af zonder een technische fouttrace.
        print('\nInvoer afgebroken. Programma afgesloten.')
