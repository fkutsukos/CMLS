############################################################
### 							 ###
### Information zur Dokumentation der Instrumentenklänge ###
### 							 ###
############################################################



###########################################
### Erklärung zur Informationskodierung ###
###########################################

Den Audiodateien wurde die relevante Information nach einem festen Schema in den Dateinamen kodiert. Diese setzt sich aus vier Teilen zusammen, die durch '-' (Bindestrich) getrennt sind.

*** Kodierungsschema monophon ***

A - Instrumententyp
	B - Bass
	G - Gitarre

B - Instrumenteneinstellung
	1 - Yamaha BB604, 1. Einstellung
	2 - Yamaha BB604, 2. Einstellung
	3 - Warwick Corvette $$, 1. Einstellung
	4 - Warwick Corvette $$, 2. Einstellung
	5 - nicht vergeben
	6 - Schecter Diamond C-1 Classic, 1. Einstellung
	7 - Schecter Diamond C-1 Classic, 2. Einstellung
	8 - Chester Stratocaster, 1. Einstellung
	9 - Chester Stratocaster, 1. Einstellung

C -  Spieltechnik
	1 - Fingeranschlag normal/leise
	2 - Fingeranschlag laut
	3 - Plektrumanschlag

D - Midinummer der Tonhöhe
	zweistellig zwischen 28 und 76

E - Saitennummer
	1 - E
	2 - A
	3 - D
	4 - G
	5 - H
	6 - E

F - Bundnummer
	zweistellig zwischen 00 und 12

G - Effektgruppe
	1 - Kein Effekt
	2 - Raumeffekt
	3 - Modulationseffekt
	4 - Verzerrungseffekt

H - Einzeleffekt
	11 - Kein Effekt
	12 - Kein Effekt, verstärkersimulation
	21 - Feedback Delay
	22 - Slapback Delay
	23 - Reverb
	31 - Chorus
	32 - Flanger
	33 - Phaser
	34 - Tremolo
	35 - Vibratp
	41 - Distortion
	42 - Overdrive

I - Effekteinstellung
	1,2 oder 3

K - Identifikationsnummer
	fünfstellig, fortlaufend mit führenden Nullen

Beispiel
B11-28100-3311-00625
ABC-DDEFF-GHHI-KKKKK
--> A-C: Bass, Yamaha, 1. Einstellung, Fingeranschlag normal/leise
    D-F: Midinr. 28, E-Saite, 0. Bund (Leersaite)--> tiefes E
    G-I: Modulationseffekt, Chorus, 1. Einstellung
    K  : Identifikationsnummer
	

*** Kodierungsschema polyphon ***

A - Instrumententyp
	G - Gitarre

B - Instrumenteneinstellung
	6 - Schecter Diamond C-1 Classic, 1. Einstellung
	9 - Chester Stratocaster, 1. Einstellung

C -  Spieltechnik
	4 - Plektrumanschlag, Intervalle
	5 - Plektrumanschlag, Drei- und Vierklänge

D - Midinummer der Tonhöhe
	zweistellig zwischen 43 und 57

E - Polyphonietyp
	11 - kleine Terz
	12 - große Terz
	13 - reine Quarte
	14 - reine Quinte
	15 - kleine Septime
	16 - große Septime
	17 - Oktave
	21 - Dur-Dreiklang
	22 - Moll-Dreiklang
	23 - Sus4-Dreiklang
	24 - Power Chord
	25 - Großer Durseptimenakkord
	26 - Kleiner Durseptimenakkord
	27 - Kleiner Mollseptimenakkord

F - nicht vergeben, immer 0

G - Effektgruppe
	1 - Kein Effekt
	2 - Raumeffekt
	3 - Modulationseffekt
	4 - Verzerrungseffekt

H - Einzeleffekt
	11 - Kein Effekt
	12 - Kein Effekt, verstärkersimulation
	21 - Feedback Delay
	22 - Slapback Delay
	23 - Reverb
	31 - Chorus
	32 - Flanger
	33 - Phaser
	34 - Tremolo
	35 - Vibratp
	41 - Distortion
	42 - Overdrive

I - Effekteinstellung
	1,2 oder 3

K - Identifikationsnummer
	fünfstellig, fortlaufend mit führenden Nullen

Beispiel
P64-43110-3311-46225
ABC-DDEEF-GHHI-KKKKK
--> A-C: Gitarre, Schecter,Plektrumanschlag Intervall
    D-F: Midinr. 43, kleine Terz
    G-I: Modulationseffekt, Chorus, 1. Einstellung
    K  : Identifikationsnummer

######################################################
### Zusammenfassen der Informationen in XML-Listen ###
######################################################

Jede Liste beeinhaltet die Informationen über alle Audiofiles von einem Instrument einer spezifischen Einstellung, die mit einem Audioeffekt in einer spezifischen Einstellung bearbeitet wurden.Jede Liste enthält einen Knoten mit Informationen zum Audioeffekt und N weitere Knoten mit Informationen zu den Audiodateien. Die Informationen sind in Kindknoten abgelegt, deren knotennamen selbsterklärend sind. Wie bei den Audiodateien ist die relevante Information in den Dateinamen der XML-Datei kodiert.

*** Kodierungsschema Listen ***

A-C - Instrumenteninformation, wie bei den Audiodateien
D-F - nicht vergeben, immer 0
G-I - Effektinformation, wie bei den Audiodateien
K   - Identifikationsnummer, dreistellig, fortlaufend mit führenden Nullen

Beispiel
B11-00000-3311-013
ABC-DDEEF-GHHI-KKK
--> A-C: Gitarre, Schecter,Fingeranschlag
    D-F: ohne Bedeutung
    G-I: Modulationseffekt, Chorus, 1. Einstellung
    K  : Identifikationsnummer
