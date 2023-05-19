TODO
- Grafiek voor detecties (Assignment 4 ofz)

- Deelopdrachten
    - Assignment 1: Corner detectie (Wachten op parameter finetuning)
    - Assignment 2: database matchen tegen zichzelf en verschil tussen twee beste resultaten bijhouden => gemiddelde nemen hiervan
    - Assignment 2: database matchen met beelden van video en verschil tussen twee beste resultaten bijhouden => gemiddelde nemen hiervan
        Hiervoor eerst alle beelden door functie van assignemnt 1 laten lopen

- Matching verbeteren
    - Gewogen som van ~20 returns nemen om zaal beter te bepalen
    - # Flann matches mee returnen om zo te bepalen of het een goeie match is
        - Totaal # matches vergelijken
        - Verhouding van # matches tov 2de beste match (kleiner verschil tov 2de bij slechte match)
    - Van alle gevonden paintings enkel de beste match gebruiken
    - Indien andere room zou zijn, de x aantal volgende frames checken of dit wel zo is, indien zo dan pas room tovoegen

- Painting detector v2
    - Parameters verbeteren => Nick
    - Achtergrond uitfilteren => Louis
        - HSV bereik meegeven met functie afh v kamer
        - Bereik uitfilteren (vervangen door wit) van frame voor betere detectie
        - Toepassen op een vekleinde foto !!!! (Duurt zeer lang), dan opnieuw vergroten en toepassen op frame
        - Beeld extraxten uit orig frame

