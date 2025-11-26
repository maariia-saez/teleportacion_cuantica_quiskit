"""
=============================================================================
CÓDIGO DE SIMULCIONES CON QISKIT
=============================================================================

Aálisis Teórico y Experimental de la Teleportación Cuántica.
Simulaciones realizadas con Qiskit (IBM).

Autores:
    - Alicia Elvira Montes Núñez
    - María Sáez Díaz
    
Fecha: Noviembre 2025

Este script implementa las tres simulaciones de teleportación cuántica
descritas en la Sección 5 del trabajo.

Simulación 1: Implementa el protocolo idea BBCJPW con correcciones.
Simulación 2: Implementa el protocolo probabilístico de 1997 (Zeilinger)
                eliminando las correcciones.
Simulación 3: Ejecuta el protocolo ideal en hardware cuántico real de IBM
                para medir el impacto del ruido.
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

# ---------------------------------------------------------------------------
# --- CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------

# Conexión al servicio de IBM Quantum (requiere tener credenciales guardadas)
service = QiskitRuntimeService()

# --- Definición del estado a teleportar ---
# En lugar de usar solo el estado |+> (que sería theta=np.pi/2), usamos
# un estado genérico |ψ> = cos(θ/2)|0> + sin(θ/2)|1>.
# Usamos theta = pi/3 como un ejemplo no tivial.
theta = np.pi / 3
initial_state_vector = [np.cos(theta / 2), np.sin(theta / 2)]

# Número de "shots" o repeticiones en cada experimento
SHOTS = 4096

# ---------------------------------------------------------------------------
# --- Simulación 1: Protocolo Ideal (BBCJPW, con correcciones)
# ---------------------------------------------------------------------------

print("\n--- Iniciando Simulación 1: Protocolo Ideal ---\n")

# --- PASO 1: Creación del Circuito ---
# Necesitamos 3 qubits:
# q[0]: El estado |ψ⟩ que Alice quiere teleportar (el "original").
# q[1]: El qubit de Alice que forma parte del par EPR.
# q[2]: El qubit de Bob, entrelazado con q[1].
qc1 = QuantumCircuit(3)

# Registros clásicos:
# c_alice (2 bits): Almacena el resultado de la Medida de Estado de Bell (BSM).
# c_bob_verif (1 bit): Almacena el resultado de la verificación final de Bob.
c_alice = ClassicalRegister(2, "alice_meas")
c_bob_verif = ClassicalRegister(1, "bob_verif")
qc1.add_register(c_alice)
qc1.add_register(c_bob_verif)

# --- PASO 2: Preparación de Estados ---
# Alice prepara q[0] en el estado |ψ⟩ que queremos enviar.
qc1.initialize(initial_state_vector, 0)
qc1.barrier()

# --- PASO 3: Creación del Par EPR (Canal Cuántico) ---
# Se genera un estado de Bell (El estado |Φ+⟩) entre q[1] (Alice) y q[2] (Bob).
# Esto debe hacerse ANTES de que Alice realice la teleportación.
qc1.h(1)
qc1.cx(1, 2)
qc1.barrier()

# --- PASO 4: Protocolo de Alice (Medida de Estado de Bell) ---
# Alice entrelaza su qubit original (q[0]) con su parte del par EPR (q[1]).
# Esta secuencia (CX seguida de H) es la implementación estándar de una BSM.
qc1.cx(0, 1)
qc1.h(0)
qc1.barrier()

# Alice mide sus dos qubits (q[0] y q[1]) y guarda el resultado (00, 01, 10 o 11)
# en su registro clásico. Esto destruye el estado original en q[0].
qc1.measure([0, 1], c_alice)
qc1.barrier()

# --- PASO 5: Protocolo de Bob (Corrección Clásica) ---
# Este es el "canal clásico". Bob espera a recibir los 2 bits de Alice.
# Dependiendo del resultado, aplica una corrección a su qubit (q[2])
# para reconstruir el estado |ψ⟩ original.
#
# Nota: La corrección depende del par EPR usado. Para |Φ+⟩ (H.CX):
# - Si Alice mide 00: Bob no hace nada (Identidad).
# - Si Alice mide 01: Bob aplica Z.
# - Si Alice mide 10: Bob aplica X.
# - Si Alice mide 11: Bob aplica Z y luego X.
#
# (La implementación if_test de Qiskit usa los bits en orden inverso,
#  por eso c_alice[0] es el bit de Z y c_alice[1] es el de X).
with qc1.if_test((c_alice[1], 1)):
    qc1.x(2) # Se aplica si el bit 1 (de q[0]) es 1
with qc1.if_test((c_alice[0], 1)):
    qc1.z(2) # Se aplica si el bit 0 (de q[1]) es 1

# --- PASO 6: Verificación ---
# ¿Cómo sabemos si funcionó? Bob aplica la operación INVERSA a la preparación
# inicial. Si q[2] es ahora |ψ⟩ = Ry(theta)|0⟩, entonces aplicarle Ry(-theta)
# debe devolverlo siempre al estado |0⟩.
qc1.ry(-theta, 2)
# Medimos el qubit de Bob. Si la teleportación fue exitosa,
# el resultado en c_bob_verif DEBE ser '0' el 100% de las veces.
qc1.measure(2, c_bob_verif)

# --- Ejecución de la Simulación 1 ---
# Usamos AerSampler, un simulador local ideal (sin ruido).
print("\n--- Simulación 1: Protocolo Ideal ---\n")
sim1_sampler = AerSampler(run_options={"shots": 4096})
result1 = sim1_sampler.run([qc1]).result()
counts1 = result1.quasi_dists[0].binary_probabilities()
print("Resultados simulación 1 (ideal):", counts1)

# --- Gráfico de la Simulación 1 ---
# Convertimos probabilidades a conteos enteros para el histograma
counts1_int = {k: int(v * SHOTS) for k, v in counts1.items()}
plt.figure(figsize=(6, 4))
plot_histogram(counts1_int, color='royalblue')
plt.title('Simulación 1: Protocolo Ideal')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# --- Simulación 2: Teleportación Probabilística (sin correcciones)
# ---------------------------------------------------------------------------
print("\n--- Iniciando Simulación 2: Probabilística (Sin Corrección) ---\n")

# --- PASO 1: Creación del Circuito ---
# Esta simulación replica la limitación del experimento de 1997:
# la incapacidad de realizar la corrección clásica (Paso 5).
qc2 = QuantumCircuit(3)
c_alice_2 = ClassicalRegister(2, "alice_meas_2")
c_bob_2 = ClassicalRegister(1, "bob_verif_2")
qc2.add_register(c_alice_2)
qc2.add_register(c_bob_2)

# --- PASOS 2, 3 y 4: (Idénticos a la Simulación 1) ---
qc2.ry(theta, 0)# PASO 2: Preparar estado
qc2.barrier()
qc2.h(1)        # PASO 3: Crear par EPR
qc2.cx(1, 2)
qc2.barrier()
qc2.cx(0, 1)    # PASO 4: BSM de Alice
qc2.h(0)
qc2.barrier()
qc2.measure([0, 1], c_alice_2)  # Medición de Alice
qc2.barrier()

# --- PASO 5: ELIMINADO ---
# Aquí simulamos la limitación experimental. Bob NO aplica ninguna
# corrección (X o Z) independientemente de lo que mida Alice.

# --- PASO 6: Verificación ---
# Bob realiza la misma verificación que antes.
qc2.ry(-theta, 2)
qc2.measure(2, c_bob_2)

# --- Ejecución de la Simulación 2 ---
# Se espera que esta simulación FALLE el 75% de las veces.
# El éxito (medir '0' en c_bob_2) solo ocurrirá si Alice, por casualidad,
# midió '00', que es el único caso que no requería corrección.
sim2_sampler = AerSampler(run_options={"shots": 4096})
result2 = sim2_sampler.run([qc2]).result()
counts2 = result2.quasi_dists[0].binary_probabilities()
print("Resultados simulación 2 (probabilística):", counts2)

# --- Gráfico de la Simulación 2 ---
counts2_int = {k: int(v * SHOTS) for k, v in counts2.items()}
plt.figure(figsize=(6, 4))
plot_histogram(counts2_int, color='darkorange')
plt.title('Simulación 2: Probabilística')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# --- Simulación 3: Hardware Real (IBM Quantum)
# ---------------------------------------------------------------------------
print("\n--- Iniciando Simulación 3: Ejecutando en Hardware Real IBM ---\n")

# --- PASO 1: Selección del Backend ---
# Se utiliza el circuito ideal (qc1), ya que el objetivo es
# ver cómo un protocolo perfecto es afectado por el ruido del mundo real.
#
# Buscamos el procesador cuántico (backend) real menos ocupado
# que tenga al menos 3 qubits.
backend_real = service.least_busy(simulator=False, operational=True, min_num_qubits=3)
print(f"Backend elegido: {backend_real.name}")

# --- PASO 2: Transpilación ---
# "Transpilar" es adaptar nuestro circuito ideal (qc1) a la arquitectura
# física del backend elegido (backend_real). Esto implica reescribir
# las puertas lógicas (ej. CNOT) a las puertas nativas que el chip
# puede ejecutar y manejar la conectividad entre qubits.
pm = generate_preset_pass_manager(backend=backend_real, optimization_level=1)
tp_qc1 = pm.run(qc1)    # tp_qc1 es el circuito "transpilado"

# --- PASO 3: Ejecución en Hardware ---
# Inicializamos el Sampler V2, esta vez apuntando al backend real.
sampler_real = Sampler(mode=backend_real)

print("Enviando circuito al hardware real...")

# Enviamos el circuito transpilado (tp_qc1) al backend.
# Esto es asíncrono: el 'job' se pone en una cola en la nube.
job3 = sampler_real.run([tp_qc1], shots=4096)
print(f"ID del job: {job3.job_id()}")
print("Esperando resultados...")
# Esta línea pausa el script hasta que el job se completa y devuelve los resultados.
result3 = job3.result()
print("Resultados recibidos")

# --- PASO 4: Procesamiento de Resultados ---
# El Sampler V2 devuelve los resultados por registro clásico.
# Extraemos los conteos del único registro que medimos al final: 'bob_verif'.
pub_result = result3[0] # Accedemos al resultado del primer (y único) circuito
print(pub_result.data.keys())
counts3 = pub_result.data['bob_verif'].get_counts()
print("Resultados hardware real:", counts3)

# Se espera un resultado similar a la Simulación 1 (casi todo '0'),
# pero con una pequeña (pero no nula) cantidad de '1' debido al
# ruido, la decoherencia y los errores en las puertas del chip.
fidelidad = counts3.get('0', 0) / SHOTS
print(f"Fidelidad observada (proporción de '0'): {fidelidad*100:.2f}%")

# --- Gráfico de la Simulación 3 ---
plt.figure(figsize=(6, 4))
plot_histogram(counts3, color='crimson')
plt.title('Simulación 3: Hardware Real IBM')
plt.tight_layout()
plt.show()

# ===========================================================================
# --- GUARDAR IMÁGENES DE LOS CIRCUITOS
# ===========================================================================

# Guardar el diagrama del circuito ideal (para Figura 1)
qc1.draw('mpl', filename='circuito_sim1.png')

# Guardar el diagrama del circuito probabilístico (para Figura 3)
qc2.draw('mpl', filename='circuito_sim2.png')

# Guardar el diagrama del circuito transpilado (para Figura 5)
# (Este se verá mucho más complejo que qc1, mostrando las puertas nativas)
tp_qc1.draw('mpl', filename='circuito_sim3.png')