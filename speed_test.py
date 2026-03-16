#!/usr/bin/env python3
"""
Confronto velocita: HiGHS (CPU) vs CUDA Simplex solver (GPU).

Esegue tutti i file .dat in data/Dati-LP/ attraverso entrambi i risolutori
piu volte, misura il tempo di esecuzione e stampa una tabella riassuntiva
con i rapporti di accelerazione.
"""

import glob
import os
import re
import statistics
import subprocess
import sys
import time

try:
    import highspy
except ImportError:
    print("ERRORE: highspy non installato. Installa con: pip install highspy")
    sys.exit(1)

# ── Configurazione ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTS_DIR = os.path.join(SCRIPT_DIR, "data", "Dati-LP")
CUDA_BINARY = os.path.join(SCRIPT_DIR, "simplex.out")

WARMUP_RUNS = 2       # iterazioni di riscaldamento (scartate)
TIMED_RUNS = 10       # iterazioni misurate per problema
TOLERANCE = 1e-4      # tolleranza per il controllo correttezza


# ═══════════════════════════════════════════════════════════════════════════════
# Risolutore HiGHS
# ═══════════════════════════════════════════════════════════════════════════════
def solve_highs(dat_path: str):
    """Risolvi con HiGHS; restituisce (valore_obiettivo, stato, tempo_sec)."""
    h = highspy.Highs()
    h.silent()
    st = h.readModel(dat_path)
    if st != highspy.HighsStatus.kOk:
        return None, "READ_ERROR", 0.0

    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0

    model_status = h.getModelStatus()
    obj = h.getInfoValue("objective_function_value")[1]

    # Mappa lo stato HiGHS a stringhe canoniche
    status_str = str(model_status)
    if "kOptimal" in status_str:
        status = "OPTIMAL"
    elif "kUnbounded" in status_str:
        status = "UNBOUNDED"
    elif "kInfeasible" in status_str:
        status = "INFEASIBLE"
    else:
        status = status_str
    return obj, status, elapsed


def bench_highs(dat_path: str, warmup: int, runs: int):
    """Benchmark HiGHS: riscaldamento, poi raccogli `runs` campioni di tempo."""
    for _ in range(warmup):
        solve_highs(dat_path)

    times = []
    obj = None
    status = None
    for _ in range(runs):
        obj, status, t = solve_highs(dat_path)
        times.append(t)
    return obj, status, times


# ═══════════════════════════════════════════════════════════════════════════════
# Risolutore CUDA Simplex
# ═══════════════════════════════════════════════════════════════════════════════
def solve_cuda(binary: str, dat_path: str):
    """Esegui CUDA simplex; restituisce (valore_obiettivo, stato, tempo_sec).

    Il tempo di esecuzione e letto dal timer interno hpc_gettime() del
    programma, che misura solo il calcolo escludendo I/O e output.
    """
    result = subprocess.run(
        [binary, "-s", dat_path],
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout

    # Parsing dello stato da stdout
    status = "UNKNOWN"
    if "Status: OPTIMAL" in output:
        status = "OPTIMAL"
    elif "Status: INFEASIBLE" in output:
        status = "INFEASIBLE"
    elif "Status: UNBOUNDED" in output:
        status = "UNBOUNDED"
    elif "Status: TIMEOUT" in output:
        status = "TIMEOUT"
    elif "Status: ERROR" in output:
        status = "ERROR"
    elif result.returncode != 0:
        status = "ERROR"

    # Parsing valore obiettivo: "Objective Value: %.6f"
    m = re.search(r"Objective Value:\s*([-\d.eE+]+)", output)
    obj = float(m.group(1)) if m else None

    # Preferisci tempo interno riportato da hpc_gettime()
    m_time = re.search(r"Elapsed time:\s*([\d.eE+-]+)\s*seconds", output)
    elapsed = float(m_time.group(1)) if m_time else None

    return obj, status, elapsed


def bench_cuda(binary: str, dat_path: str, warmup: int, runs: int):
    """Benchmark CUDA simplex: riscaldamento, poi raccogli `runs` campioni."""
    for _ in range(warmup):
        solve_cuda(binary, dat_path)

    times = []
    obj = None
    status = None
    for _ in range(runs):
        obj, status, t = solve_cuda(binary, dat_path)
        if t is not None:
            times.append(t)
    return obj, status, times


# ═══════════════════════════════════════════════════════════════════════════════
# Helper per la stampa
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_ms(sec: float) -> str:
    """Formatta secondi come stringa millisecondi."""
    return f"{sec * 1000:.3f}"


def stats_summary(times: list) -> dict:
    """Restituisce media, mediana, min, max, dev_std di una lista di secondi."""
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.isfile(CUDA_BINARY):
        print(f"ERRORE: Binario CUDA non trovato in {CUDA_BINARY}")
        sys.exit(1)

    # Accetta directory o file individuali come argomenti CLI
    # Default: data/Dati-LP/
    targets = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_TESTS_DIR]
    dat_files = []
    for t in targets:
        if os.path.isdir(t):
            dat_files.extend(glob.glob(os.path.join(t, "*.dat")))
        elif os.path.isfile(t) and t.endswith(".dat"):
            dat_files.append(t)
        else:
            print(f"AVVISO: salto {t} (non e' una directory o file .dat)")
    dat_files = sorted(dat_files)
    if not dat_files:
        print(f"Nessun file .dat trovato in: {targets}")
        sys.exit(1)

    print("=" * 90)
    print("  TEST VELOCITA: HiGHS (CPU) vs CUDA Simplex (GPU)")
    print(f"  Esecuzioni riscaldamento: {WARMUP_RUNS}  |  Esecuzioni misurate: {TIMED_RUNS}")
    print("=" * 90)

    # Intestazione
    hdr = (
        f"{'Problema':<20} "
        f"{'HiGHS (ms)':>12} {'CUDA (ms)':>12} {'Speedup':>10} "
        f"{'Match':>7} {'Stato':>10}"
    )
    print(f"\n{hdr}")
    print("-" * 90)

    results = []

    for dat_path in dat_files:
        name = os.path.splitext(os.path.basename(dat_path))[0]
        sys.stdout.write(f"  {name:<18} ")
        sys.stdout.flush()

        # ── Benchmark HiGHS ──
        h_obj, h_status, h_times = bench_highs(dat_path, WARMUP_RUNS, TIMED_RUNS)
        h_stats = stats_summary(h_times)

        # ── Benchmark CUDA ──
        c_obj, c_status, c_times = bench_cuda(CUDA_BINARY, dat_path, WARMUP_RUNS, TIMED_RUNS)
        if not c_times:
            print(f"{'FALLITO':>12} {'FALLITO':>12} {'N/A':>10} {'FALLITO':>7} {c_status:>10}")
            continue
        c_stats = stats_summary(c_times)

        # ── Controllo correttezza ──
        if h_obj is not None and c_obj is not None:
            match = abs(h_obj - c_obj) < TOLERANCE and h_status == c_status
        else:
            match = h_status == c_status

        # ── Speedup (tempo HiGHS / tempo CUDA; >1 significa CUDA piu veloce) ──
        if c_stats["mean"] > 0:
            speedup = h_stats["mean"] / c_stats["mean"]
        else:
            speedup = float("inf")

        row = {
            "name": name,
            "h_mean": h_stats["mean"],
            "c_mean": c_stats["mean"],
            "speedup": speedup,
            "match": match,
            "status": h_status,
            "h_stats": h_stats,
            "c_stats": c_stats,
            "h_obj": h_obj,
            "c_obj": c_obj,
        }
        results.append(row)

        # Stampa riga compatta
        sp_str = f"{speedup:.2f}x" if speedup != float("inf") else "inf"
        print(
            f"{fmt_ms(h_stats['mean']):>12} "
            f"{fmt_ms(c_stats['mean']):>12} "
            f"{sp_str:>10} "
            f"{'OK' if match else 'FALLITO':>7} "
            f"{h_status:>10}"
        )

    # ── Statistiche dettagliate ──
    print("\n" + "=" * 90)
    print("  STATISTICHE DETTAGLIATE (millisecondi)")
    print("=" * 90)

    detail_hdr = (
        f"{'Problema':<20} {'Risolutore':<10} "
        f"{'Media':>10} {'Mediana':>10} {'Min':>10} {'Max':>10} {'DevStd':>10}"
    )
    print(f"\n{detail_hdr}")
    print("-" * 90)

    for r in results:
        hs, cs = r["h_stats"], r["c_stats"]
        print(
            f"{r['name']:<20} {'HiGHS':<10} "
            f"{fmt_ms(hs['mean']):>10} {fmt_ms(hs['median']):>10} "
            f"{fmt_ms(hs['min']):>10} {fmt_ms(hs['max']):>10} {fmt_ms(hs['stdev']):>10}"
        )
        print(
            f"{'':<20} {'CUDA':<10} "
            f"{fmt_ms(cs['mean']):>10} {fmt_ms(cs['median']):>10} "
            f"{fmt_ms(cs['min']):>10} {fmt_ms(cs['max']):>10} {fmt_ms(cs['stdev']):>10}"
        )

    # ── Valori obiettivo ──
    print("\n" + "=" * 90)
    print("  VALORI OBIETTIVO")
    print("=" * 90)
    print(f"\n{'Problema':<20} {'HiGHS Obj':>16} {'CUDA Obj':>16} {'Diff':>12} {'Match':>7}")
    print("-" * 75)
    for r in results:
        h_o = r["h_obj"]
        c_o = r["c_obj"]
        h_str = f"{h_o:.6f}" if h_o is not None else "N/A"
        c_str = f"{c_o:.6f}" if c_o is not None else "N/A"
        if h_o is not None and c_o is not None:
            diff = abs(h_o - c_o)
            d_str = f"{diff:.2e}"
        else:
            d_str = "N/A"
        print(f"{r['name']:<20} {h_str:>16} {c_str:>16} {d_str:>12} {'OK' if r['match'] else 'FALLITO':>7}")

    # ── Riepilogo ──
    total_highs = sum(r["h_mean"] for r in results)
    total_cuda = sum(r["c_mean"] for r in results)
    total_speedup = total_highs / total_cuda if total_cuda > 0 else float("inf")
    all_match = all(r["match"] for r in results)

    print("\n" + "=" * 90)
    print("  RIEPILOGO")
    print("=" * 90)
    print(f"  Problemi testati:     {len(results)}")
    print(f"  Tutti i risultati ok: {'SI' if all_match else 'NO'}")
    print(f"  Tempo totale HiGHS:   {fmt_ms(total_highs)} ms")
    print(f"  Tempo totale CUDA:    {fmt_ms(total_cuda)} ms")
    sp_str = f"{total_speedup:.2f}x" if total_speedup != float("inf") else "inf"
    faster = "CUDA" if total_speedup > 1.0 else "HiGHS"
    print(f"  Speedup complessivo:  {sp_str} ({faster} piu veloce)")
    print("=" * 90)

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
