## Algoritmo del Simplesso Primale

**Step 1. Inizializzazione:**
Definisce una soluzione base ammissibile $\mathbf{x} = [\mathbf{x}_B, \mathbf{x}_N] = [\mathbf{B}^{-1}\mathbf{b}, \mathbf{0}] = [\bar{\mathbf{b}}, \mathbf{0}]$ di costo $z = \mathbf{c}_B\mathbf{x}_B = \mathbf{c}_B\mathbf{B}^{-1}\mathbf{b}$.

**Step 2. Pricing:**
Calcola $\mathbf{w} = \mathbf{c}_B\mathbf{B}^{-1}$, che equivale a risolvere $\mathbf{wB} = \mathbf{c}_B$.
Calcola i costi ridotti $\mathbf{wa}_j - c_j$ per le variabili non-base $j \in N$ e determina:
$$\mathbf{wa}_k - c_k = \max_{j \in N} \{\mathbf{wa}_j - c_j\}$$

**Step 3. Condizioni di ottimalità:**
Se $\mathbf{wa}_k - c_k < 0$, allora STOP: la soluzione è *ottima*.

**Step 4. La variabile $k$ è candidata a entrare in base:**
Calcola $\mathbf{y}^k = \mathbf{B}^{-1}\mathbf{a}_k$, che equivale a risolvere $\mathbf{B}\mathbf{y}^k = \mathbf{a}_k$.
Se $\mathbf{y}^k \leq \mathbf{0}$, allora STOP: la soluzione è *illimitata*.

---

## Algoritmo del Simplesso Primale (2)

**Step 5. Rapporto minimo:**
Calcola il valore da assegnare a $x_k$:
$$x_k = \frac{\bar{b}_r}{y^k_r} = \min \left\{ \frac{\bar{b}_i}{y^k_i} : y^k_i > 0, i = 1, \dots, m \right\}$$

La variabile $x_r$ esce dalla base e $x_k$ entra al suo posto.
Aggiorna $\mathbf{B}$, $\mathbf{N}$ e la soluzione base $\mathbf{x} = [\mathbf{x}_B, \mathbf{x}_N] = [\bar{\mathbf{b}}, \mathbf{0}]$.
Ritorna allo Step 2.

---

## Il Metodo del Simplesso in Formato Tableau

Il simplesso primale in formato tableau semplifica le operazioni di aggiornamento della base, della corrispondente soluzione e dei costi ridotti $\mathbf{wa}_j - c_j$ ad ogni iterazione.

Dato il problema in forma standard:
$$\min z = \mathbf{c}_B\mathbf{x}_B + \mathbf{c}_N\mathbf{x}_N$$
$$\mathbf{Bx}_B + \mathbf{Nx}_N = \mathbf{b}$$
$$\mathbf{x}_B, \mathbf{x}_N \geq \mathbf{0}$$

si può riscrivere come:
$$\min z$$
$$z - \mathbf{c}_B\mathbf{x}_B - \mathbf{c}_N\mathbf{x}_N = 0$$
$$\mathbf{x}_B + \mathbf{B}^{-1}\mathbf{Nx}_N = \mathbf{B}^{-1}\mathbf{b}$$
$$\mathbf{x}_B, \mathbf{x}_N \geq \mathbf{0}$$

Moltiplicando la seconda equazione per $\mathbf{c}_B$ e sommandola alla prima si ottiene:
$$\min z$$
$$z + \mathbf{0x}_B + (\mathbf{c}_B\mathbf{B}^{-1}\mathbf{N} - \mathbf{c}_N)\mathbf{x}_N = \mathbf{c}_B\mathbf{B}^{-1}\mathbf{b}$$
$$\mathbf{x}_B + \mathbf{B}^{-1}\mathbf{Nx}_N = \mathbf{B}^{-1}\mathbf{b}$$
$$\mathbf{x}_B, \mathbf{x}_N \geq \mathbf{0}$$

Il risultato può essere inserito in un *tableau* come segue:

|   | $z$ | $\mathbf{x}_B$ | $\mathbf{x}_N$ | RHS |
|---|-----|-----------------|-----------------|-----|
| $z$ | $1$ | $\mathbf{0}$ | $\mathbf{c}_B\mathbf{B}^{-1}\mathbf{N} - \mathbf{c}_N$ | $\mathbf{c}_B\mathbf{B}^{-1}\mathbf{b}$ |
| $\mathbf{x}_B$ | $0$ | $\mathbf{I}$ | $\mathbf{B}^{-1}\mathbf{N}$ | $\mathbf{B}^{-1}\mathbf{b}$ |

dove il Right Hand Side (RHS) contiene il valore della funzione obiettivo e delle variabili base.

---

## Metodo 2-Fasi

Sia dato un problema della seguente forma:

$$(P) \quad z_P = \min \mathbf{cx}$$
$$\text{s.t.} \quad \mathbf{Ax} = \mathbf{b}$$
$$\mathbf{x} \geq \mathbf{0}$$

Nell'ipotesi che $\mathbf{b} \geq \mathbf{0}$, si possono aggiungere $m$ variabili $\mathbf{x}_A$, dette *artificiali*, alle $n$ variabili originarie, e risolvere il seguente problema:

$$(P') \quad z_{P'} = \min \mathbf{1x}_A$$
$$\text{s.t.} \quad \mathbf{Ax} + \mathbf{Ix}_A = \mathbf{b}$$
$$\mathbf{x}, \mathbf{x}_A \geq \mathbf{0}$$

dove $\mathbf{I}$ è la matrice identità di ordine $m$ e $\mathbf{1} = \{1, 1, \dots, 1\}$ è un vettore di $m$ componenti tutte pari a 1.

---

## Metodo 2-Fasi (2)

In questo caso i problemi $P$ e $P'$ non sono equivalenti.
Risolvere il problema $P'$ serve solo a determinare una soluzione base ammissibile per il problema $P$.

Sia $(\mathbf{x}^*, \mathbf{x}_A^*)$ la soluzione ottima del problema $P'$ di valore $z_{P'}$. Si possono presentare tre casi:

1. $z_{P'} > 0$: il problema $P$ **non ha una base ammissibile**.
2. $z_{P'} = 0$ e nessuna variabile artificiale è in base: il problema $P$ **ha una base ammissibile**.
3. $z_{P'} = 0$ e almeno una variabile artificiale è in base: il problema $P$ **ha una base ammissibile, ma bisogna *estrarla***.

---

## Metodo 2-Fasi (3)

Se $z_{P'} = 0$ e una variabile artificiale è in base, per generare una base senza variabili artificiali è necessario farla uscire. Questo caso si verifica quando la soluzione è *degenere*, ossia una variabile in base ha valore nullo.

Sia la riga $i$ del tableau corrispondente alla variabile artificiale $x_h^A$ in base con valore $\bar{b}_i = 0$:

- Se esiste un $y_i^j \neq 0$ (per qualche $j = 1, \dots, n$), allora possiamo **pivotare** su questo coefficiente e la variabile $x_j$ entra in base al posto della variabile artificiale $x_h^A$.
- Se $y_i^j = 0$ per ogni $j = 1, \dots, n$, allora possiamo eliminare dal tableau sia la riga $i$ che la colonna della variabile artificiale $x_h^A$.