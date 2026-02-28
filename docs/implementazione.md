PIANO DI IMPLEMENTAZIONE: SIMPLESSO PRIMALE (METODO 2 FASI) IN CUDA

1. GESTIONE DELLA MEMORIA E STRUTTURE DATI
   - Host (CPU): Inizializzazione della matrice dei vincoli, vettore dei costi, 
     vettore dei termini noti.
   - Device (GPU): Allocazione di un singolo array 1D contiguo per rappresentare 
     l'intero Tableau (matrice appiattita). Questo è fondamentale in CUDA per 
     garantire la coalescenza della memoria durante le letture/scritture.
   - Stato: Vettore di interi per tracciare gli indici delle variabili in base.

2. LOGICA DELLE DUE FASI (Orchestrata dall'Host)
   - FASE 1 (Ricerca base ammissibile):
     * Aggiunta logica delle variabili artificiali al Tableau.
     * Sostituzione della riga della funzione obiettivo originale con la 
       minimizzazione della somma delle variabili artificiali.
     * Esecuzione del Ciclo del Simplesso (lancio dei kernel).
     * Verifica: se all'ottimo la funzione obiettivo > 0 (considerando un epsilon 
       di tolleranza), il problema è inammissibile.
     * Rimozione/Mascheramento delle colonne artificiali.
   - FASE 2 (Ricerca dell'ottimo):
     * Ripristino dei costi originali nella riga della funzione obiettivo 
       (aggiornati rispetto alla base corrente).
     * Esecuzione del Ciclo del Simplesso fino a condizione di ottimalità.

3. DESIGN DEI KERNEL CUDA (La parte parallela)
   - Kernel A: Trovare la Colonna Pivot (Variabile Entrante)
     * Utilizzo del pattern di "Parallel Reduction" (Riduzione Parallela) sulla 
       riga dei costi ridotti per trovare il valore minimo (se negativo).
   - Kernel B: Trovare la Riga Pivot (Variabile Uscente)
     * Ogni thread calcola in parallelo il Minimum Ratio Test (b[i] / A[i][pivot_col]).
     * Nuova "Parallel Reduction" per trovare l'indice con il rapporto minimo positivo.
   - Kernel C: Aggiornamento del Tableau (Pivotazione)
     * È l'operazione più costosa e più parallelizzabile.
     * Mappatura 2D dei thread: ogni thread (o blocco di thread) è responsabile 
       dell'aggiornamento di una o più celle del Tableau.
     * Ottimizzazione: caricare la riga pivot e la colonna pivot nella Shared 
       Memory del blocco, in quanto verranno lette da tutti i thread.

4. FLUSSO DI ESECUZIONE (Host-Device Loop)
   - cudaMalloc (allocazione su GPU).
   - cudaMemcpy (trasferimento Dati Iniziali da CPU a GPU).
   - CICLO WHILE:
     1. Lancia Kernel A. Se non ci sono costi negativi -> Ottimo trovato. Break.
     2. Lancia Kernel B. Se nessun rapporto è valido -> Problema Illimitato. Break.
     3. Lancia Kernel C per aggiornare la matrice.
     4. cudaDeviceSynchronize() per assicurarsi che l'aggiornamento sia finito.
   - cudaMemcpy (recupero Soluzione Finale da GPU a CPU).
   - cudaFree (pulizia memoria).