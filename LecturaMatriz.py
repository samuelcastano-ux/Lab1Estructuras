# -*- coding: utf-8 -*-
"""
===============================================================================
 demo.py — Programa principal: crea, verifica y manipula la matriz,
 y MIDE el uso de RAM y el tiempo, para demostrar que los tres problemas
 del enunciado quedaron resueltos.
===============================================================================

Uso:
    python demo.py                # usa N = 100.000 (matriz de 1.25 GB)
    python demo.py --n 8000       # prueba rápida con una matriz más chica
    python demo.py --rehacer      # fuerza recrear el archivo aunque ya exista
"""

import argparse
import os
import time
import tracemalloc

import numpy as np

import crearmatriz as md


def medir(nombre_paso, funcion, *args, **kwargs):
    """
    Ejecuta `funcion` midiendo:
      - tiempo transcurrido (segundos)
      - pico de memoria RAM asignada por el PROCESO PYTHON durante el paso
        (tracemalloc mide asignaciones de objetos Python/numpy en RAM; el
        memmap en sí no cuenta como asignación porque vive en el archivo,
        que es justamente el punto: demostrar que la RAM usada es poca e
        independiente del tamaño total de la matriz).
    Imprime un resumen y retorna lo que haya devuelto `funcion`.
    """
    tracemalloc.start()
    t0 = time.time()
    resultado = funcion(*args, **kwargs)
    t1 = time.time()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[BENCHMARK] {nombre_paso}: {t1 - t0:.2f} s, "
          f"pico de RAM en Python = {pico / (1024 * 1024):.2f} MB\n")
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Demo de matriz binaria en disco")
    parser.add_argument("--n", type=int, default=md.N, help="tamaño de la matriz (múltiplo de 8)")
    parser.add_argument("--ruta", type=str, default="matriz.bin", help="archivo de salida")
    parser.add_argument("--bloque-mb", type=int, default=md.BLOQUE_MB, help="RAM por bloque, en MB")
    parser.add_argument("--rehacer", action="store_true", help="recrear el archivo aunque ya exista")
    args = parser.parse_args()

    n = args.n
    ruta = args.ruta
    esperado = n * (n // 8)

    print("=" * 70)
    print(f"MATRIZ BINARIA DE {n:,} x {n:,} BITS  (~{esperado / 1024**3:.3f} GB en disco)")
    print("=" * 70)

    # -------------------------------------------------------------------
    # PASO 1: creación (o reutilización si ya existe con el tamaño correcto)
    # -------------------------------------------------------------------
    if args.rehacer or not os.path.exists(ruta) or os.path.getsize(ruta) != esperado:
        medir(
            "Creación de la matriz",
            md.crear_matriz, ruta, n, md.SEMILLA, args.bloque_mb, True,
        )
    else:
        print(f"Archivo '{ruta}' ya existe con el tamaño correcto; no se recrea.\n")

    mm = md.abrir_matriz(ruta, n)

    # -------------------------------------------------------------------
    # PASO 2: informe de verificación completo
    # -------------------------------------------------------------------
    medir("Verificación de integridad", md.informe, mm, ruta, md.SEMILLA)

    # -------------------------------------------------------------------
    # PASO 3: ejemplos de lectura (para mostrar que son rápidos y baratos)
    # -------------------------------------------------------------------
    print("=" * 70)
    print("EJEMPLOS DE LECTURA")
    print("=" * 70)

    fila_ejemplo = min(50, n - 1)
    col_ejemplo = min(12345, n - 1)
    print(f"\n-- Celda puntual [{fila_ejemplo}, {col_ejemplo}] --")
    valor = medir("  leer_celda", md.leer_celda, mm, fila_ejemplo, col_ejemplo)
    print(f"   valor = {valor}")

    print("-- Primeros 20 bits de la fila 0 --")
    print("  ", medir("  leer_segmento_fila", md.leer_segmento_fila, mm, 0, 0, 20))

    print("-- Esquina superior izquierda 8x8 --")
    esquina = medir("  leer_submatriz 8x8", md.leer_submatriz, mm, 0, 8, 0, 8)
    for fila in esquina:
        print("    ", " ".join(str(b) for b in fila))

    print("-- Primeros 10 bits de la columna 0 (acceso disperso, más lento) --")
    print("  ", medir("  leer_columna (10 filas)", md.leer_columna, mm, 0, 0, 10))

    print("-- Conteo de unos en la fila 0 (usando tabla de popcount) --")
    unos = medir("  contar_unos_por_fila", md.contar_unos_por_fila, mm, 0)
    print(f"   unos = {unos:,} de {n:,} bits")

    # -------------------------------------------------------------------
    # PASO 4: manipulación (modificar una celda y confirmar el cambio)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EJEMPLO DE MANIPULACIÓN (escritura puntual)")
    print("=" * 70)

    del mm  # cerrar el mapeo de solo lectura antes de reabrir en r+
    mm_rw = md.abrir_matriz(ruta, n, escritura=True)

    f_obj, c_obj = min(50_000, n - 1), min(80_005, n - 1)
    original = md.leer_celda(mm_rw, f_obj, c_obj)
    nuevo = 1 - original
    medir("  escribir_celda", md.escribir_celda, mm_rw, f_obj, c_obj, nuevo)
    confirmado = md.leer_celda(mm_rw, f_obj, c_obj)
    print(f"Celda [{f_obj:,}, {c_obj:,}]: {original} -> {confirmado} "
          f"({'OK' if confirmado == nuevo else 'FALLA'})")

    del mm_rw
    print("\nListo.")


if __name__ == "__main__":
    main()