TODO
- Grafiek voor detecties (Assignment 4 ofz)

- Upkuisen van matrixen voor camera

- Deelopdrachten
    - Assignment 1: Corner detectie (Wachten op parameter finetuning)
    - Assignment 2: database matchen tegen zichzelf en verschil tussen twee beste resultaten bijhouden => gemiddelde nemen hiervan
    - Assignment 2: database matchen met beelden van video en verschil tussen twee beste resultaten bijhouden => gemiddelde nemen hiervan
        Hiervoor eerst alle beelden door functie van assignemnt 1 laten lopen

- Matching verbeteren => Nick en Louis
    - Andere matcher dan knnMatcher proberen
    - Gewogen som van ~20 returns nemen om zaal beter te bepalen
    - # Flann matches mee returnen om zo te bepalen of het een goeie match is
        - Totaal # matches vergelijken
        - Verhouding van # matches tov 2de beste match (kleiner verschil tov 2de bij slechte match)
    - Van alle gevonden paintings enkel de beste match gebruiken
    - Indien andere room zou zijn, de x aantal volgende frames checken of dit wel zo is, indien zo dan pas room tovoegen

- Painting detector v2
    - Parameters verbeteren => Nick
    - Zandloper effect bij 5 punten bekijken => Nick
    - Rand filter verbeteren => Nick (Mss oppervlakte er bij betrekken)
    - Kader in kader filteren
    - Room voor HSV filter

 - Paper
    - Deel 4 afwerken (Arne). Ondertussen ok denk ik? Eens nalezen?
    - Nalezen van paper + eventuele opmerkingen die Bram kan verwerken (iedereen)
    - Resultaten ass1 
    - Resultaten ass2
    - Volledig resultaat van bevindingen
