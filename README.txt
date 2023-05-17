TODO
- cameraMatrix en distCoeffs voor alle camera's / lenzen
- Painting detector verbeteren bij kleine foto's => Parameters aanpassen
    - Multi threading toepassen => Kijken welke functies het langste duren

- Grafiek voor detecties (Assignment 4 ofz)

- Deelopdrachten
    - Assignment 1: Corner detectie (Wachten op parameter finetuning)
    - Assignment 2: ... 

- Matching verbeteren
    - Gewogen som van ~20 returns nemen om zaal beter te bepalen
    - # Flann matches mee returnen om zo te bepalen of het een goeie match is
        - Totaal # matches vergelijken
        - Verhouding van # matches tov 2de beste match (kleiner verschil tov 2de bij slechte match)
    - Van alle gevonden paintings enkel de beste match gebruiken
