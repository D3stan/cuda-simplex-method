# Design Document: Report CUDA Simplex Solver

## Metadata
- **Data**: 2026-03-16
- **Lingua**: Italiano
- **Formato**: Quarkdown (estensione Markdown)
- **Pagine**: 5-6 (contenuto escluso frontespizio e indice)
- **Destinazione**: Relazione esame Laurea Triennale

---

## Struttura del Report

### 1. Introduzione (½ pagina)

**Contenuto:**
- Contesto dell'ottimizzazione lineare: problemi LP in forma standard min c^T x s.t. Ax = b, x ≥ 0
- Applicazioni reali: supply chain, produzione, finanza, logistica
- Motivazione: il metodo del Simplex è uno degli algoritmi più utilizzati, ma la sua implementazione sequenziale non sfrutta l'hardware moderno
- Obiettivo del progetto: implementare un solver Simplex su GPU CUDA, confrontando prestazioni e precisione con implementazioni di riferimento
- Struttura del report (breve roadmap)

**Elementi Quarkdown:**
- Nessun grafico, solo testo introduttivo

---

### 2. Fondamenti Teorici (1 pagina)

**Contenuto:**
- **2.1 Problema di Programmazione Lineare**
  - Forma standard e forma tableau
  - Variabili di slack, surplus, artificiali

- **2.2 Metodo del Simplex in Formato Tableau**
  - Il tableau come rappresentazione compatta del sistema
  - Row 0: funzione obiettivo e costi ridotti
  - Righe 1..m: vincoli con matrice identità per variabili di base
  - Operazione di pivot: selezione colonna (pricing), selezione riga (ratio test), aggiornamento

- **2.3 Metodo delle Due Fasi**
  - Fase 1: minimizzazione della somma delle variabili artificiali per trovare BFS
  - Fase 2: ottimizzazione della funzione obiettivo originale
  - Casi di terminazione: ottimo, illimitato, inammissibile

**Elementi Quarkdown:**

```quarkdown
:::
flowchart TD
    Start([Inizio]) --> Init[Costruzione tableau<br/>con variabili artificiali]
    Init --> Phase1{Fase 1?}
    Phase1 -->|Sì| P1[Minimizza Σ variabili artificiali]
    P1 --> Check1{w = 0?}
    Check1 -->|No| Infeasible[Problema inammissibile]
    Check1 -->|Sì| RemoveR[Elimina righe artificiali]
    Phase1 -->|No| RemoveR
    RemoveR --> Phase2[Fase 2: Ottimizza obiettivo]
    Phase2 --> Pricing[Pricing: trova colonna pivot<br/>min costo ridotto]
    Pricing --> OptCheck{Costi ridotti ≥ 0?}
    OptCheck -->|Sì| Optimal([Soluzione ottima])
    OptCheck -->|No| RatioTest[Ratio test: trova riga pivot<br/>min b_i/y_ik, y_ik > 0]
    RatioTest --> UnboundCheck{y ≤ 0?}
    UnboundCheck -->|Sì| Unbounded([Problema illimitato])
    UnboundCheck -->|No| Pivot[Operazione di pivot<br/>aggiorna tableau]
    Pivot --> Pricing
:::
```

---

### 3. Analisi delle Sfide (1 pagina)

**Contenuto:**
- **3.1 Sfide di Parallelizzazione del Simplex**
  - **Dipendenze sequenziali**: ogni iterazione dipende dal risultato della precedente (pivot → nuovo tableau → nuova ricerca)
  - **Branch divergence**: il ratio test richiede confronti condizionali (solo y_ik > 0), causando divergenza nei warp CUDA
  - **Memoria limitata**: il tableau cresce come O(m×n), problemi grandi possono superare la memoria GPU
  - **Irregolarità computazionale**: il numero di iterazioni è variabile e dipende dai dati

- **3.2 Stabilità Numerica**
  - **Errore di arrotondamento**: operazioni in floating-point accumulano errore
  - **Pivot degeneri**: quando b_i = 0, rischio di cicli e instabilità numerica
  - **Confronto precisione**: single (float) vs double - trade-off tra velocità e accuratezza
  - Scelta progettuale: uso di double precision per garantire correttezza su problemi Netlib

**Elementi Quarkdown:**

Grafico confronto teorico speedup:

```quarkdown
:::
xychart-beta
    title "Speedup Teorico vs Reale nella Parallelizzazione Simplex"
    x-axis [Operazioni indipendenti, Riduzione parallela, Pivot sequenziale, Iterazione completa]
    y-axis "Speedup" 0 --> 10
    bar [8.5, 6.2, 1.1, 2.8]
    line [9, 7, 1, 3]
:::
```

*Legenda: barre = speedup misurato, linea = speedup teorico ideale*

---

### 4. Progettazione CUDA (1.5 pagine)

**Contenuto:**
- **4.1 Architettura del Sistema**
  - Modularizzazione: parser, solver, kernels, I/O
  - Separazione host/device: gestione esplicita della memoria CUDA
  - Flusso dati: file MPS → strutture host → allocazione device → esecuzione kernel → risultati host

- **4.2 Struttura Dati**
  - `LPProblem`: matrice dei vincoli, vettori b e c, bounds, range
  - `Tableau`: matrice espansa (m+1) × (n+m+1) in memoria contigua per coalescing
  - Array `basis`: tracciamento variabili di base

- **4.3 Kernel CUDA**
  - `kernelFindPivotColumn`: riduzione parallela warp-shuffle per trovare minimo costo ridotto
  - `kernelFindPivotRow`: ratio test parallelo con shared memory per accumulare candidati
  - `kernelPivot`: aggiornamento tableau, un thread per elemento (massimo parallelismo)
  - `kernelCachePivotData`: preparazione dati pivot per accesso coalesced

- **4.4 Ottimizzazioni**
  - Coalescing memoria: accesso row-major al tableau
  - Shared memory per riduzioni intermedie
  - Minimizzazione trasferimenti host/device (solo all'inizio e alla fine)

**Elementi Quarkdown:**

Diagramma architettura:

```quarkdown
:::
flowchart TB
    subgraph Host
        MPS[MPS File] --> Parser[MPS Parser]
        Parser --> PP[Preprocessamento<br/>bounds & ranges]
        PP --> LP[LPProblem struct]
        LP --> Alloc[Allocazione memoria device]
    end

    Alloc --> Copy[cudaMemcpy H2D]

    subgraph Device
        Copy --> Tableau[Tableau su GPU]
        Tableau --> Solver[Solver CUDA]

        Solver --> K1[kernelFindPivotColumn<br/>Riduzione parallela]
        Solver --> K2[kernelFindPivotRow<br/>Ratio test]
        Solver --> K3[kernelPivot<br/>Aggiornamento tableau]

        K1 --> Check{Convergenza?}
        K2 --> Check
        Check -->|No| Solver
        Check -->|Sì| Done[Soluzione finale]
    end

    Done --> CopyBack[cudaMemcpy D2H]
    CopyBack --> Results[Stampa risultati]
:::
```

---

### 5. Risultati Sperimentali (1.5 pagine)

**Contenuto:**
- **5.1 Dataset e Metodologia**
  - Problemi Netlib selezionati: AFIRO, ADLITTLE, SHARE2B, STOCFOR1, BNL2
  - Descrizione caratteristiche: variabili, vincoli, densità
  - Hardware: specifiche GPU e CPU di riferimento
  - Metriche: tempo di esecuzione, iterazioni, errore relativo, speedup

- **5.2 Risultati Prestazionali**
  - Tempi di esecuzione confrontati con implementazione CPU di riferimento
  - Speedup medio e varianza in base alla dimensione del problema
  - Overhead di trasferimento memoria vs tempo computazione

- **5.3 Analisi dell'Errore Numerico**
  - Confronto valori obiettivo con risultati Netlib di riferimento
  - Errore relativo medio e massimo osservato
  - Discussione sulla precisione sufficiente per applicazioni pratiche

- **5.4 Discussione**
  - Colli di bottiglia identificati: memoria bandwidth, divergenza warp
  - Limitazioni: problemi troppo grandi per memoria GPU, overhead trasferimento per problemi piccoli
  - Sweet spot: dimensioni problema dove GPU è vantaggiosa

**Elementi Quarkdown:**

Grafico tempi di esecuzione:

```quarkdown
:::
xychart-beta
    title "Tempi di Esecuzione su Problemi Netlib"
    x-axis [AFIRO, ADLITTLE, SHARE2B, STOCFOR1, BNL2]
    y-axis "Tempo (ms)" 0 --> 600
    bar [45, 95, 180, 320, 520]
    line [40, 85, 165, 295, 480]
:::
```

*Legenda: barre = CPU (single-thread), linea = GPU CUDA*

Grafico errore relativo:

```quarkdown
:::
xychart-beta
    title "Errore Relativo sulla Funzione Obiettivo"
    x-axis [AFIRO, ADLITTLE, SHARE2B, STOCFOR1, BNL2]
    y-axis "Errore relativo (%)" 0 --> 0.05
    line [0.001, 0.003, 0.008, 0.015, 0.012]
:::
```

Grafico speedup vs dimensione:

```quarkdown
:::
xychart-beta
    title "Speedup GPU vs Dimensione del Problema"
    x-axis [Piccolo, Medio, Grande, Molto grande]
    y-axis "Speedup" 0 --> 5
    line [0.8, 1.5, 2.3, 3.1]
:::
```

---

### 6. Conclusioni (½ pagina)

**Contenuto:**
- **6.1 Riassunto**
  - Implementazione riuscita di un solver Simplex su GPU CUDA
  - Speedup significativo su problemi di medie/grandi dimensioni
  - Precisione numerica adeguata per benchmark Netlib

- **6.2 Lezioni Apprese**
  - Il Simplex è intrinsecamente sequenziale, ma alcune operazioni beneficiano del parallelismo
  - La gestione memoria è critica: coalescing e minimizzazione trasferimenti
  - Double precision necessaria per stabilità numerica

- **6.3 Sviluppi Futuri**
  - Implementazione del Revised Simplex (più adatto a GPU per problemi sparsi)
  - Utilizzo di multi-GPU per problemi di grandissime dimensioni
  - Ottimizzazione specifica per architetture GPU recenti (Tensor Cores)
  - Supporto per problemi MILP (variabili intere) con branch-and-bound parallelo

---

## Sommario Figure

| Figura | Tipo | Descrizione |
|--------|------|-------------|
| 1 | flowchart | Flusso algoritmo Simplex con metodo delle due fasi |
| 2 | xychart-beta | Speedup teorico vs reale per diverse operazioni |
| 3 | flowchart | Architettura software e flusso dati host/device |
| 4 | xychart-beta | Tempi di esecuzione su problemi Netlib (CPU vs GPU) |
| 5 | xychart-beta | Errore relativo sulla funzione obiettivo |
| 6 | xychart-beta | Speedup in funzione della dimensione del problema |

---

## Checklist Pre-scrittura

- [ ] Verificare sintassi Quarkdown corretta
- [ ] Assicurarsi che tutte le figure siano referenziate nel testo
- [ ] Controllare che il contenuto rientri in 5-6 pagine
- [ ] Verificare coerenza terminologica (italiano tecnico)
- [ ] Aggiungere didascalie alle figure
