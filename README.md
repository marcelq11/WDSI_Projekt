# WDSI_Projekt
Marcel Chudecki 144480
Projekt zaliczeniowy przedmiotu wprowadzenie do sztucznej inteligencji "detekcja znaków ograniczenia prędkości.
Projekt niestety posiada tylko część klasyfikacji znaków.
Podczas klasyfikacji znaków program wykonuje następujące czynności:
-wczytanie danych z wejścia (load_from_input)
-program ładuje dane treningowe z folderu train(load_from_xml)
-następuje uczeniu (learn)
-wykorzystując dane z uczenia program ekstraktuje cechy z danych obrazów
-po wyuczeniu praogem przystępuje do ekstrakcji danych wejsciowych po czym wypisuję je na wyjściu

Odnośnie części do detekcji napisałem tylko i wyłącznie funkcję do ładowania zdjęć. W funkcji main jest if, który umożliwia wybór 'classify' oraz 'detect',ale działa tylko 'classify'.
Wstępnie projekt na obecnym etapie chciałbym już oddać.
