# Analisi della Frobenius Metric su Reti Neurali

## Introduzione
Questo progetto analizza la variazione della **Frobenius Metric** per diversi tipi di reti neurali (SBM, ER, RGR, BA) in base alla soglia \( t \) e alla costante di accoppiamento (03, 05, 08, 1). Lo scopo è comprendere come la norma di Frobenius della differenza tra la matrice stimata e la matrice originale cambi al variare di questi parametri.

## Dati e Metodo
I dati sono stati ottenuti a partire da matrici di coefficiente stimate e matrici di adiacenza originali, moltiplicate per la costante di accoppiamento. Il calcolo della **Frobenius Metric** è stato fatto applicando diverse soglie alla matrice stimata e confrontandola con la matrice modificata per ottenere la norma di Frobenius della differenza normalizzata.

### Costanti di accoppiamento analizzate
Le costanti di accoppiamento analizzate sono:
- **03**
- **05**
- **08**
- **1**

## Risultati

### Grafici della Frobenius Metric
Qui sotto sono riportati i grafici ottenuti per ogni tipo di rete con diverse costanti di accoppiamento.

#### 1. SBM
![SBM - C=03](Grafici_Frobenius/Frobenius_SBM_C03.png)
![SBM - C=05](Grafici_Frobenius/Frobenius_SBM_C05.png)
![SBM - C=08](Grafici_Frobenius/Frobenius_SBM_C08.png)
![SBM - C=1](Grafici_Frobenius/Frobenius_SBM_C1.png)

#### 2. ER
![ER - C=03](Grafici_Frobenius/Frobenius_ER_C03.png)
![ER - C=05](Grafici_Frobenius/Frobenius_ER_C05.png)
![ER - C=08](Grafici_Frobenius/Frobenius_ER_C08.png)
![ER - C=1](Grafici_Frobenius/Frobenius_ER_C1.png)

#### 3. RGR
![RGR - C=03](Grafici_Frobenius/Frobenius_RGR_C03.png)
![RGR - C=05](Grafici_Frobenius/Frobenius_RGR_C05.png)
![RGR - C=08](Grafici_Frobenius/Frobenius_RGR_C08.png)
![RGR - C=1](Grafici_Frobenius/Frobenius_RGR_C1.png)

#### 4. BA
![BA - C=03](Grafici_Frobenius/Frobenius_BA_C03.png)
![BA - C=05](Grafici_Frobenius/Frobenius_BA_C05.png)
![BA - C=08](Grafici_Frobenius/Frobenius_BA_C08.png)
![BA - C=1](Grafici_Frobenius/Frobenius_BA_C1.png)

## Discussione
Osservando i grafici, notiamo che:
- **[Punto di analisi 1]**: Ad esempio, le reti SBM mostrano una variazione più sensibile della Frobenius Metric al variare della costante di accoppiamento.
- **[Punto di analisi 2]**: Le reti ER mantengono una Frobenius Metric costante fino ad un certo valore di soglia \( t \) per tutte le costanti di accoppiamento.

## Conclusione
I risultati suggeriscono che l'influenza della costante di accoppiamento sulla norma di Frobenius varia significativamente a seconda della topologia della rete. Questo potrebbe avere implicazioni nell'interpretazione della connettività delle reti neurali in relazione ai parametri di soglia.

## Note
- I grafici sono stati generati automaticamente e salvati nella cartella `Grafici_Frobenius`.
- Ogni grafico mostra la **Frobenius Metric** in funzione della soglia \( t \), limitata al range di interesse \( t \leq 1.5 \).

---

### 2. Aggiunta dei file al repository

Assicurati di avere tutti i grafici nella cartella `Grafici_Frobenius` e poi aggiungili al tuo repository GitHub insieme al file `README.md`. Esegui i seguenti comandi:

```bash
git add Grafici_Frobenius/*.png README.md
git commit -m "Aggiunta dei risultati e del report con i grafici della Frobenius Metric"
git push
