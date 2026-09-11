"""Concept opdracht 1: procesroute tonen, geen medische voorspelling.

Uitvoeren: python opdracht1/procesmodel.py
Bron en modelaannames staan in ../../informatie/opdracht1/README.md.
"""

STAPPEN = {
    'S01': 'Binnenkomst via huisarts of spoedeisende hulp',
    'S02': 'Urgente opname: verpleegkundigen verzorgen eerste zorg en informatie',
    'D01': 'Ernstig geval met evident hartinfarct?',
    'S03': 'Direct naar intensive care',
    'S04': 'Specialisten meten patiëntwaarden (meestal enkele uren)',
    'S05': 'Artsenteam beoordeelt gegevens en stelt diagnose op basis van ervaring en studies',
    'D02': 'Vervolgroute volgens gekozen scenario (criteria niet beschreven)',
    'S06': 'Verblijf op intensive care',
    'S07': 'Reguliere ziekenhuisopname',
    'S08': 'Ziekenhuis direct verlaten',
    'S09': 'Operatie, bijvoorbeeld plaatsing van stents',
    'E01': 'Overlijden (tijdstip in het proces onbekend)',
}


def modelleer_proces(instroom, ernstig, vervolg, operatie=False):
    """Geef een route en losse gebeurtenissen terug uit handmatige invoer.

    `ernstig` betekent hier exact: ernstig geval met evident hartinfarct.
    `vervolg` is ic, regulier, vertrek of onbekend_bij_overlijden.
    Een onbekende vervolgroute voorkomt dat we bij overlijden verblijf verzinnen.
    """
    if instroom not in ('huisarts', 'seh'):
        raise ValueError('Instroom moet huisarts of seh zijn.')
    if type(ernstig) is not bool or type(operatie) is not bool:
        raise ValueError('Ernstig en operatie moeten True of False zijn.')
    vervolgstappen = {
        'ic': 'S06', 'regulier': 'S07', 'vertrek': 'S08',
        'onbekend_bij_overlijden': None,
    }
    if vervolg not in vervolgstappen:
        raise ValueError('Onbekende vervolgroute.')

    route = ['S01', 'S02', 'D01']
    if ernstig:
        route.append('S03')
    route.extend(['S04', 'S05', 'D02'])
    if vervolgstappen[vervolg] is not None:
        route.append(vervolgstappen[vervolg])

    # Deze gebeurtenissen krijgen geen verzonnen plek in de tijdlijn.
    gebeurtenissen = []
    if operatie:
        gebeurtenissen.append('S09')
    if vervolg == 'onbekend_bij_overlijden':
        gebeurtenissen.append('E01')
    return route, gebeurtenissen


def toon_scenario(naam, instroom, ernstig, vervolg, operatie=False):
    route, gebeurtenissen = modelleer_proces(instroom, ernstig, vervolg, operatie)
    print(f'\n{naam} | instroom: {instroom} | ernstig: {ernstig}')
    for stap in route:
        print(f'  {stap}: {STAPPEN[stap]}')
    for stap in gebeurtenissen:
        print(f'  Losse gebeurtenis, tijdstip onbekend - {stap}: {STAPPEN[stap]}')


if __name__ == '__main__':
    # Zelfbedachte routevoorbeelden, geen patiëntgegevens of medische adviezen.
    toon_scenario('Voorbeeld A', 'huisarts', False, 'vertrek')
    toon_scenario('Voorbeeld B', 'seh', True, 'ic', operatie=True)
    toon_scenario('Voorbeeld C', 'huisarts', False, 'regulier')
    toon_scenario('Voorbeeld D', 'seh', True, 'onbekend_bij_overlijden')
