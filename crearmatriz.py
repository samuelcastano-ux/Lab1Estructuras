# -*- coding: utf-8 -*-
"""
===============================================================================
 matriz_disco.py
 Matriz binaria de N x N bits almacenada en disco duro (memory-mapped file)
===============================================================================

PROBLEMA A RESOLVER
--------------------
Se necesita crear, almacenar, leer y manipular una matriz binaria de
100.000 x 100.000 bits (10.000.000.000 de bits = 1.25 GB). Si se intentara
crear esa matriz como un array normal de numpy (o peor, como una lista de
Python de 0s y 1s), el programa necesitaría muchísima más RAM de la real
(cada "0"/"1" como int64 de numpy ocuparía 8 bytes -> 80 GB de RAM) y además
sería muy lento porque no habría forma de sincronizar eso con el disco de a
poquitos.

Este módulo resuelve tres problemas concretos que pedía el enunciado:

1) CONSUMO EXCESIVO DE RAM
   -> La matriz completa NUNCA se carga en memoria. Se usa `numpy.memmap`,
      que "mapea" el archivo del disco a un array de numpy: cuando se lee o
      escribe una porción, el sistema operativo mueve solo esa porción entre
      disco y RAM (a través de su caché de páginas), no el archivo entero.
   -> Además, se empaqueta la matriz en bits reales (8 bits lógicos por cada
      byte físico) con `np.packbits` / `np.unpackbits`, en vez de usar un
      byte (o más) por cada 0/1. Esto reduce el tamaño en disco (y por tanto
      lo que hay que mover) en un factor de 8.
   -> Tanto la creación como la lectura se hacen SIEMPRE por bloques de
      tamaño acotado (parámetro `bloque_mb`), calculado para que el bloque
      ocupe una cantidad fija de megabytes sin importar qué tan grande sea N.
      Así el consumo de RAM del proceso se mantiene aproximadamente
      constante, aunque N crezca.

2) ESCRITURA LENTA A DISCO
   -> La versión original llamaba a `mm.flush()` después de CADA bloque.
      `flush()` obliga al sistema operativo a sincronizar inmediatamente esa
      porción con el disco físico (I/O síncrona), lo cual es muy costoso si
      se repite miles de veces. Aquí se deja que el sistema operativo use su
      caché de escritura normalmente (I/O asíncrona/perezosa) y solo se hace
      UN flush al final (más `os.fsync` para forzar la sincronización real
      con el hardware). Esto reduce drásticamente el número de operaciones
      de I/O síncronas.
   -> El tamaño de bloque se calibra para que cada escritura sea una
      operación grande y contigua en disco (más eficiente que muchas
      escrituras pequeñas), en vez de una fila a la vez.

3) OPTIMIZACIÓN EN MANIPULACIÓN, CREACIÓN, ALMACENAMIENTO Y LECTURA
   -> Todas las operaciones están vectorizadas con numpy (nada de bucles
      `for` bit a bit en Python).
   -> Para leer o escribir UN SOLO bit no se desempaqueta la fila completa:
      se calcula directamente a qué byte y qué posición dentro del byte
      pertenece ese bit, y se opera con máscaras binarias (`&`, `|`, `~`)
      sobre ese único byte.
   -> Para leer un rango de columnas dentro de una fila, solo se leen y
      desempaquetan los bytes que "cubren" ese rango (no la fila entera).
   -> Se documenta explícitamente por qué leer una COLUMNA completa es la
      operación más costosa de todas (acceso disperso / no contiguo en
      disco) y se ofrece una forma más eficiente de hacerlo por bloques de
      filas cuando se necesitan varias columnas a la vez.

===============================================================================
"""

from __future__ import annotations

import os
import hashlib
import numpy as np


# =============================================================================
# 0. PARÁMETROS POR DEFECTO
# =============================================================================

N = 100_000                # Filas y columnas LÓGICAS (bits) de la matriz
BYTES_FILA = N // 8         # Bytes físicos que ocupa cada fila en disco
RUTA_POR_DEFECTO = "matriz.bin"
SEMILLA = 42                 # Semilla del generador aleatorio (reproducibilidad)
BLOQUE_MB = 32                # Cuántos MB de RAM puede usar cada bloque de trabajo


# =============================================================================
# 1. UTILIDADES DE DIMENSIONAMIENTO DE BLOQUES
# =============================================================================

def filas_por_bloque(bytes_fila: int, bloque_mb: int = BLOQUE_MB) -> int:
    """
    Calcula cuántas filas caben en un bloque de `bloque_mb` megabytes.

    Esta es la pieza clave contra el "consumo excesivo de RAM": en vez de
    fijar un número de filas a mano (que para N pequeño gastaría poca RAM,
    pero para N grande podría gastar demasiada), se calcula el número de
    filas a partir de un presupuesto de memoria FIJO. Así el uso de RAM del
    programa no depende de qué tan grande sea la matriz.

    Parámetros
    ----------
    bytes_fila : int
        Bytes físicos que ocupa una fila (N // 8).
    bloque_mb : int
        Presupuesto de memoria por bloque, en megabytes.

    Retorna
    -------
    int : número de filas por bloque (al menos 1).
    """
    bytes_bloque = bloque_mb * 1024 * 1024
    filas = max(1, bytes_bloque // bytes_fila)
    return int(filas)


# =============================================================================
# 2. CREACIÓN DE LA MATRIZ
# =============================================================================

def crear_matriz(
    ruta: str = RUTA_POR_DEFECTO,
    n: int = N,
    semilla: int = SEMILLA,
    bloque_mb: int = BLOQUE_MB,
    verbose: bool = True,
) -> str:
    """
    Crea en disco el archivo binario de la matriz de n x n bits, generando
    y escribiendo los datos por bloques para no exceder un presupuesto fijo
    de RAM ni saturar el disco con escrituras síncronas.

    Parámetros
    ----------
    ruta : str
        Nombre del archivo a crear (se sobrescribe si ya existe).
    n : int
        Filas/columnas lógicas (bits). Debe ser múltiplo de 8.
    semilla : int
        Semilla del generador aleatorio. Con la misma semilla el archivo
        generado es siempre idéntico bit a bit (clave para verificar).
    bloque_mb : int
        Presupuesto de RAM por bloque de escritura (ver `filas_por_bloque`).
    verbose : bool
        Si True, imprime el progreso y estadísticas finales.

    Retorna
    -------
    str : la ruta del archivo creado.
    """
    if n % 8 != 0:
        raise ValueError(f"n debe ser múltiplo de 8 para empaquetar bits (n={n})")

    bytes_fila = n // 8
    bloque = filas_por_bloque(bytes_fila, bloque_mb)

    # mode="w+": crea el archivo del tamaño EXACTO (n * bytes_fila) de una
    # sola vez (el sistema de archivos lo reserva como archivo disperso) y
    # permite leer/escribir sobre él a través del mapeo de memoria.
    mm = np.memmap(ruta, dtype=np.uint8, mode="w+", shape=(n, bytes_fila))
    rng = np.random.default_rng(semilla)

    import time
    t0 = time.time()

    for i in range(0, n, bloque):
        filas = min(bloque, n - i)
        # Generación vectorizada: un byte aleatorio = 8 bits aleatorios de
        # la matriz. Esto evita generar bit a bit (8x menos llamadas al
        # generador aleatorio) y se asigna de golpe a la porción mapeada.
        mm[i:i + filas] = rng.integers(0, 256, size=(filas, bytes_fila), dtype=np.uint8)

        # A propósito NO se llama a mm.flush() aquí: hacerlo en cada
        # iteración forzaría una sincronización síncrona con el disco en
        # cada bloque, que es precisamente el cuello de botella que
        # queremos evitar. Se deja que el sistema operativo administre la
        # caché de escritura y se sincroniza todo UNA sola vez al final.
        if verbose and (i // bloque) % 20 == 0:
            print(f"  escritas {i + filas:,} / {n:,} filas", end="\r")

    # Sincronización final única: mm.flush() vuelca la caché de numpy/mmap,
    # y os.fsync() le pide al sistema operativo que confirme la escritura
    # física en el dispositivo de almacenamiento.
    mm.flush()
    fd = os.open(ruta, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    del mm

    t1 = time.time()
    if verbose:
        tam = os.path.getsize(ruta)
        vel = (tam / (1024 * 1024)) / (t1 - t0) if t1 > t0 else float("inf")
        print(f"\nArchivo '{ruta}' creado: {tam:,} bytes en {t1 - t0:.2f} s "
              f"({vel:.1f} MB/s, bloque={bloque:,} filas / {bloque_mb} MB)")
    return ruta


def abrir_matriz(ruta: str = RUTA_POR_DEFECTO, n: int = N, escritura: bool = False):
    """
    Abre el archivo ya creado como un memmap (no carga nada en RAM todavía).

    Parámetros
    ----------
    ruta : str
        Ruta del archivo binario.
    n : int
        Filas/columnas lógicas de la matriz (debe coincidir con la creación).
    escritura : bool
        Si False (por defecto), se abre en modo "r" (solo lectura), lo que
        protege el archivo de modificaciones accidentales. Si True, se abre
        en modo "r+" (lectura y escritura) para poder modificar celdas.

    Retorna
    -------
    np.memmap de forma (n, n // 8): n filas, n//8 columnas de BYTES.
    Importante: mm.shape[1] son BYTES, no bits. Para columnas lógicas use
    `dimensiones(mm)["columnas_bits"]`.
    """
    modo = "r+" if escritura else "r"
    return np.memmap(ruta, dtype=np.uint8, mode=modo, shape=(n, n // 8))


# =============================================================================
# 3. LECTURA EFICIENTE (sin desempaquetar más de lo necesario)
# =============================================================================

def leer_celda(mm, fila: int, columna: int) -> int:
    """
    Lee UN SOLO bit. No usa np.unpackbits (que desempaquetaría todo un
    byte para quedarse con 1 bit); en su lugar calcula la posición exacta
    del bit dentro del byte y la extrae con una máscara. Es la forma más
    barata posible de leer una celda.
    """
    byte_col = columna // 8
    desplazamiento = 7 - (columna % 8)   # el bit 0 es el más significativo
    byte_valor = int(mm[fila, byte_col])
    return (byte_valor >> desplazamiento) & 1


def escribir_celda(mm, fila: int, columna: int, valor: int) -> None:
    """
    Escribe UN SOLO bit sin tocar los otros 7 bits del byte. Igual que en
    `leer_celda`, se opera directamente sobre el byte con máscaras en vez
    de desempaquetar y volver a empaquetar la fila completa.

    Requiere que `mm` se haya abierto con `abrir_matriz(..., escritura=True)`.
    """
    byte_col = columna // 8
    desplazamiento = 7 - (columna % 8)
    mascara = 1 << desplazamiento

    byte_actual = int(mm[fila, byte_col])
    if valor:
        byte_nuevo = byte_actual | mascara
    else:
        byte_nuevo = byte_actual & (~mascara & 0xFF)

    mm[fila, byte_col] = byte_nuevo
    # flush selectivo: solo de esta escritura puntual, no de todo el archivo.
    mm.flush()


def leer_fila(mm, fila: int) -> np.ndarray:
    """
    Devuelve la fila completa como array de bits (0/1). Cuesta una sola
    lectura contigua de `bytes_por_fila` bytes (rápida: las filas están
    almacenadas de forma contigua en disco).
    """
    return np.unpackbits(np.asarray(mm[fila]))


def leer_segmento_fila(mm, fila: int, inicio: int, fin: int) -> np.ndarray:
    """
    Devuelve los bits de una fila en el rango de columnas [inicio, fin),
    desempaquetando solo los bytes que cubren ese rango (no la fila
    entera). Útil para inspeccionar un tramo de una fila muy larga.
    """
    byte_ini = inicio // 8
    byte_fin = (fin + 7) // 8
    bits = np.unpackbits(np.asarray(mm[fila, byte_ini:byte_fin]))
    desfase = inicio - byte_ini * 8
    return bits[desfase:desfase + (fin - inicio)]


def leer_columna(mm, columna: int, inicio: int = 0, fin: int | None = None) -> np.ndarray:
    """
    Devuelve una columna completa (o un tramo) como array de bits.

    ADVERTENCIA DE RENDIMIENTO: una columna toca UN byte de cada fila, y
    las filas están separadas `bytes_por_fila` bytes en disco (aquí,
    12.500). Leer una columna obliga al sistema a "saltar" por todo el
    archivo en vez de leer algo contiguo, así que es órdenes de magnitud
    más lento que leer una fila. Si se necesitan MUCHAS columnas a la vez,
    es mucho más eficiente usar `leer_submatriz` (lee un bloque de filas
    de una sola vez y extrae varias columnas de ese bloque en memoria) en
    lugar de llamar a `leer_columna` una vez por columna.
    """
    if fin is None:
        fin = mm.shape[0]
    byte_col = columna // 8
    posicion = 7 - (columna % 8)
    columna_bytes = np.asarray(mm[inicio:fin, byte_col])
    return ((columna_bytes >> posicion) & 1).astype(np.uint8)


def leer_submatriz(mm, fila_ini: int, fila_fin: int, col_ini: int, col_fin: int) -> np.ndarray:
    """
    Devuelve un bloque rectangular de bits como matriz 2D, en una sola
    operación vectorizada (una lectura contigua de filas + un
    `np.unpackbits` por lote, en vez de un bucle Python fila por fila).
    """
    byte_ini = col_ini // 8
    byte_fin = (col_fin + 7) // 8
    bloque_bytes = np.asarray(mm[fila_ini:fila_fin, byte_ini:byte_fin])
    bloque_bits = np.unpackbits(bloque_bytes, axis=1)
    desfase = col_ini - byte_ini * 8
    return bloque_bits[:, desfase:desfase + (col_fin - col_ini)]


def contar_unos_por_fila(mm, fila: int) -> int:
    """
    Cuenta cuántos bits en 1 tiene una fila SIN desempaquetarla bit a bit:
    usa una tabla de "popcount" (cuántos unos tiene cada valor de byte
    posible, 0..255) precalculada una sola vez, y la aplica de forma
    vectorizada sobre los bytes de la fila. Es mucho más rápido que
    `leer_fila(...).sum()` porque no genera 100.000 elementos intermedios.
    """
    tabla = _tabla_popcount()
    return int(tabla[np.asarray(mm[fila])].sum())


_TABLA_POPCOUNT_CACHE = None


def _tabla_popcount() -> np.ndarray:
    """Calcula (una sola vez, y la reutiliza) la tabla de popcount 0..255."""
    global _TABLA_POPCOUNT_CACHE
    if _TABLA_POPCOUNT_CACHE is None:
        _TABLA_POPCOUNT_CACHE = np.unpackbits(
            np.arange(256, dtype=np.uint8).reshape(-1, 1), axis=1
        ).sum(axis=1).astype(np.uint16)
    return _TABLA_POPCOUNT_CACHE


# =============================================================================
# 4. VERIFICACIÓN DE INTEGRIDAD
# =============================================================================

def dimensiones(mm) -> dict:
    """Dimensiones REALES de la matriz, deducidas del propio memmap."""
    filas, bytes_fila = mm.shape
    return {
        "filas": filas,
        "bytes_por_fila": bytes_fila,
        "columnas_bits": bytes_fila * 8,
        "bits_totales": filas * bytes_fila * 8,
        "bytes_totales": filas * bytes_fila,
    }


def verificar_tamano_archivo(ruta: str, n: int) -> dict:
    """Compara el tamaño esperado del archivo contra el tamaño real en disco."""
    esperado = n * (n // 8)
    real = os.path.getsize(ruta)
    return {
        "ok": esperado == real,
        "bytes_esperados": esperado,
        "bytes_reales": real,
        "gb_reales": round(real / 1024 ** 3, 3),
    }


def verificar_no_vacia(mm, cuantas: int = 30, semilla: int = 4) -> dict:
    """
    Comprueba (por muestreo, sin leer el archivo completo) que no hay filas
    completamente en cero, señal típica de un bloque que nunca se escribió.
    """
    rng = np.random.default_rng(semilla)
    filas = rng.integers(0, mm.shape[0], size=cuantas)
    vacias = [int(f) for f in filas if not np.asarray(mm[int(f)]).any()]
    return {"ok": len(vacias) == 0, "filas_vacias_encontradas": vacias}


def verificar_reproducibilidad(
    mm, semilla: int = SEMILLA, filas_a_probar: int = 3
) -> dict:
    """
    Verificación fuerte de integridad: regenera EXACTAMENTE las primeras
    `filas_a_probar` filas con la misma semilla que se usó en `crear_matriz`
    y las compara byte a byte contra lo que hay en disco. Si coinciden, el
    archivo no está corrupto ni desalineado. No hace falta releer el
    archivo completo: solo se regeneran y comparan unas pocas filas.
    """
    bytes_fila = mm.shape[1]
    rng = np.random.default_rng(semilla)
    esperado = rng.integers(0, 256, size=(filas_a_probar, bytes_fila), dtype=np.uint8)
    real = np.asarray(mm[0:filas_a_probar])
    coincide = bool(np.array_equal(esperado, real))
    return {"ok": coincide, "filas_comparadas": filas_a_probar}


def checksum_bloque(mm, fila_ini: int, fila_fin: int) -> str:
    """
    Calcula un hash SHA-256 de un bloque de filas. Sirve para comparar dos
    matrices (o dos versiones del mismo archivo en distintos momentos) sin
    tener que traer el bloque entero a un formato Python: se hashea
    directamente el buffer de bytes.
    """
    datos = np.asarray(mm[fila_ini:fila_fin]).tobytes()
    return hashlib.sha256(datos).hexdigest()


def informe(mm, ruta: str, semilla: int = SEMILLA) -> bool:
    """
    Corre todas las verificaciones y las imprime de forma legible. Es la
    función a llamar para responder "¿quedó bien la matriz?".
    """
    def marca(ok):
        return "OK   " if ok else "FALLA"

    print("=" * 70)
    print("INFORME DE VERIFICACIÓN DE LA MATRIZ")
    print("=" * 70)

    d = dimensiones(mm)
    print("\n[1] DIMENSIONES")
    print(f"    Filas .................. {d['filas']:,}")
    print(f"    Columnas (bits) ........ {d['columnas_bits']:,}")
    print(f"    Bytes por fila ......... {d['bytes_por_fila']:,}")
    print(f"    Bits totales ........... {d['bits_totales']:,}")
    print(f"    Bytes totales .......... {d['bytes_totales']:,}")

    t = verificar_tamano_archivo(ruta, d["filas"])
    print(f"\n[2] TAMAÑO DEL ARCHIVO EN DISCO ....... {marca(t['ok'])}")
    print(f"    Esperado: {t['bytes_esperados']:,} bytes")
    print(f"    Real:     {t['bytes_reales']:,} bytes  ({t['gb_reales']} GB)")

    r = verificar_reproducibilidad(mm, semilla)
    print(f"\n[3] REPRODUCIBILIDAD (misma semilla) .. {marca(r['ok'])}")
    print(f"    Filas comparadas contra regeneración: {r['filas_comparadas']}")

    v = verificar_no_vacia(mm)
    print(f"\n[4] SIN FILAS EN BLANCO (muestreo) ..... {marca(v['ok'])}")
    print(f"    Filas vacías encontradas: {v['filas_vacias_encontradas']}")

    todo_ok = all([t["ok"], r["ok"], v["ok"]])
    print("\n" + "=" * 70)
    print("RESULTADO GLOBAL:", "TODO CORRECTO" if todo_ok else "HAY FALLAS")
    print("=" * 70)
    return todo_ok