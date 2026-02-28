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