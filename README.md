# Laboratorio 1 — Matriz binaria de 100.000 x 100.000 bits en disco duro

## Objetivo

Diseñar e implementar un programa que cree, almacene, lea y manipule una
matriz binaria de **100.000 x 100.000 bits** (10.000.000.000 de bits ≈
1,16 GB) usando el disco duro como almacenamiento principal, resolviendo:

1. El **consumo excesivo de RAM**.
2. La **escritura lenta a disco**.
3. La **optimización** en la manipulación, creación, almacenamiento y
   lectura de los datos.

## Archivos del repositorio

| Archivo | Descripción |
| :------ | :---------- |
| `matriz_disco.py` | Librería principal: creación, apertura, lectura, escritura y verificación de la matriz. Todo el código está documentado en español, función por función. |
| `demo.py` | Programa principal / demo. Crea la matriz (o reutiliza el archivo si ya existe), corre el informe de verificación, muestra ejemplos de lectura y manipulación, y **mide tiempo y RAM real usados en cada paso**. |
| `requirements.txt` | Dependencias (`numpy`). |
| `.gitignore` | Excluye el archivo binario generado (`*.bin`); no se versiona porque es reproducible a partir del código y la semilla. |

## Cómo ejecutar

```bash
pip install -r requirements.txt

# Matriz completa (100.000 x 100.000, ~1.16 GB en disco)
python demo.py

# Prueba rápida con una matriz más pequeña (para revisar el código sin
# esperar la generación completa)
python demo.py --n 8000 --ruta prueba.bin

# Forzar recrear el archivo aunque ya exista
python demo.py --rehacer
```

## Diseño general

La matriz **nunca existe completa en RAM**. Se almacena en disco como un
archivo binario donde cada bit lógico de la matriz es un bit físico real
(empaquetado 8 bits por byte con `np.packbits`/`np.unpackbits`), y se accede
a través de `numpy.memmap`: un array de numpy "mapeado" directamente al
archivo, de forma que leer o escribir una porción mueve solo esa porción
entre disco y RAM, nunca el archivo completo.

`
Bit lógico [fila, columna]  --->  byte físico = columna // 8
                                   posición dentro del byte = columna % 8
`

## Cómo se resolvió cada problema

### 1. Consumo excesivo de RAM

- La matriz se genera y se lee **siempre por bloques**, nunca de una sola
  vez. El tamaño del bloque no se fija en número de filas "a ojo": se
  calcula a partir de un **presupuesto de RAM en megabytes**
  (`filas_por_bloque` en `matriz_disco.py`), de modo que el uso de memoria
  del proceso Python se mantiene aproximadamente constante sin importar
  qué tan grande sea `N`.
- Se usa empaquetado real de bits (1 bit lógico = 1 bit físico), en vez de
  1 byte (o más) por cada 0/1, lo que reduce el tamaño en disco —y todo lo
  que hay que mover entre disco y RAM— en un factor de 8.
- Operaciones puntuales (leer o escribir una sola celda) tocan **un solo
  byte**, no la fila ni la matriz completa.

**Medido en esta máquina:** crear la matriz completa de 1,16 GB usó un pico
de solo **~33 MB de RAM** en Python (medido con `tracemalloc`).

### 2. Escritura lenta a disco

- La versión original llamaba a `mm.flush()` después de cada bloque
  escrito, lo cual fuerza una sincronización síncrona con el disco en cada
  iteración (muy costoso si se repite miles de veces). En este diseño solo
  se hace **un `flush()` + `os.fsync()` al final** de toda la escritura; el
  sistema operativo administra la caché de escritura de forma normal
  mientras tanto.
- El tamaño de bloque (por defecto 32 MB) hace que cada escritura sea una
  operación grande y contigua en disco, en vez de escribir fila por fila.

**Medido en esta máquina:** la matriz completa (1,16 GB) se generó y
escribió en disco en **~6,3 segundos** (~190 MB/s).

### 3. Optimización en manipulación, creación, almacenamiento y lectura

- Todo está vectorizado con numpy: no hay bucles de Python bit a bit.
- `leer_celda` / `escribir_celda`: operan sobre un único byte con máscaras
  binarias (`&`, `|`, `~`), sin desempaquetar/empaquetar la fila completa.
- `leer_segmento_fila` y `leer_submatriz`: desempaquetan solo los bytes que
  cubren el rango pedido, no la fila o el bloque completo.
- `contar_unos_por_fila`: usa una tabla de *popcount* (bits en 1 de cada
  valor de byte 0–255, precalculada una sola vez) en vez de desempaquetar
  bit a bit.
- `leer_columna` está documentada explícitamente como la operación más
  costosa (acceso disperso: toca un byte de cada una de las 100.000 filas,
  separadas 12.500 bytes entre sí), y se recomienda `leer_submatriz` cuando
  se necesitan varias columnas a la vez, para leer un bloque de filas de
  una sola vez en lugar de recorrer el archivo completo por cada columna.

## Cómo se verifica que la matriz quedó bien

`demo.py` llama a `informe()`, que corre y reporta:

1. **Dimensiones reales** del archivo (deducidas del propio `memmap`, no de
   constantes fijas en el código).
2. **Tamaño del archivo en disco** contra el tamaño teórico esperado
   (`N * N // 8` bytes).
3. **Reproducibilidad**: se regeneran en memoria las primeras filas con la
   misma semilla (`SEMILLA = 42`) usada al crear el archivo y se comparan
   byte a byte contra lo que hay en disco. Si coinciden, el archivo no está
   corrupto ni desalineado.
4. **Ausencia de filas en blanco**: se muestrean filas al azar y se
   confirma que ninguna quedó completamente en cero (síntoma típico de un
   bloque que nunca llegó a escribirse).

Además, `matriz_disco.checksum_bloque(mm, fila_ini, fila_fin)` permite
calcular un hash SHA-256 de cualquier bloque de filas, útil para comparar
dos copias del archivo (por ejemplo, antes y después de moverlo o
comprimirlo) sin tener que leerlo entero a Python.

## Resultado de una corrida real (100.000 x 100.000)

Archivo 'matriz.bin' creado: 1,250,000,000 bytes en 6.27 s (190.1 MB/s)
[BENCHMARK] Creación de la matriz: 6.29 s, pico de RAM en Python = 33.31 MB

[2] TAMAÑO DEL ARCHIVO EN DISCO ....... OK
[3] REPRODUCIBILIDAD (misma semilla) .. OK
[4] SIN FILAS EN BLANCO (muestreo) ..... OK

RESULTADO GLOBAL: TODO CORRECTO
