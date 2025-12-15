# %% [MD]
# =============================================================================
# CÓDIGO DE SIMULACIONES CON QISKIT
# =============================================================================
# Análisis Teórico y Experimental de la Teleportación Cuántica.
# Simulaciones realizadas con Qiskit (IBM).
#
# Autores:
#    - Alicia Elvira Montes Núñez
#    - María Sáez Díaz
#    
# Fecha: Noviembre 2025

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from IPython.display import display

r

print("Librerías importadas y listas.")

# %% [MD]
# --- 1. CONFIGURACIÓN GLOBAL ---
# Conexión al servicio de IBM Quantum y preparación de estados.

# %%
# Conexión al servicio de IBM Quantum
try:
    service = QiskitRuntimeService()
    print("Conexión con IBM Quantum establecida.")
except:
    print("Aviso: No se detectó cuenta de IBM Quantum guardada. La Simulación 3 no se ejecutará correctamente.")

# --- Definición del estado a teleportar ---
# Usamos theta = pi/3 (ejemplo no trivial)
theta = np.pi / 3
initial_state_vector = [np.cos(theta / 2), np.sin(theta / 2)]
SHOTS = 4096

print(f"Estado preparado para teleportar con theta = {theta:.2f}")

# %% [MD]
# --- 2. SIMULACIÓN 1: PROTOCOLO IDEAL ---
# Asumimos hardware perfecto y correcciones clásicas completas.

# %%
print("\n--- Ejecutando Simulación 1: Protocolo Ideal ---")

qc1 = QuantumCircuit(3)
c_alice = ClassicalRegister(2, "alice_meas")
c_bob_verif = ClassicalRegister(1, "bob_verif")
qc1.add_register(c_alice, c_bob_verif)

# 1. Preparación del estado
qc1.initialize(initial_state_vector, 0)
qc1.barrier()

# 2. Creación del Par EPR (Canal Cuántico)
qc1.h(1)
qc1.cx(1, 2)
qc1.barrier()

# 3. Medida de Bell (Alice)
qc1.cx(0, 1)
qc1.h(0)
qc1.barrier()
qc1.measure([0, 1], c_alice)
qc1.barrier()

# 4. Correcciones Clásicas (Bob)
with qc1.if_test((c_alice[1], 1)):
    qc1.x(2)
with qc1.if_test((c_alice[0], 1)):
    qc1.z(2)

# 5. Verificación (Operación inversa)
qc1.ry(-theta, 2)
qc1.measure(2, c_bob_verif)

# Ejecución en simulador ideal
sim1_sampler = AerSampler(run_options={"shots": SHOTS})
result1 = sim1_sampler.run([qc1]).result()
counts1 = result1.quasi_dists[0].binary_probabilities()

# Visualización de resultados
counts1_int = {k: int(v * SHOTS) for k, v in counts1.items()}
print(f"Resultados Ideales: {counts1_int}")

fig1 = plot_histogram(counts1_int, color='royalblue', title='Simulación 1: Protocolo Ideal')
display(fig1)

# Visualización del circuito
print("Esquema del Circuito 1:")
display(qc1.draw('mpl'))

# %% [MD]
# --- 3. SIMULACIÓN 2: EXPERIMENTO DE 1997 ---
# Simulamos la limitación histórica eliminando las correcciones.

# %%
print("\n--- Ejecutando Simulación 2: Zeilinger 1997 ---")

qc2 = QuantumCircuit(3)
c_alice_2 = ClassicalRegister(2, "alice_meas_2")
c_bob_2 = ClassicalRegister(1, "bob_verif_2")
qc2.add_register(c_alice_2, c_bob_2)

# Pasos idénticos a la Simulación 1 (Preparación, EPR, BSM)...
qc2.ry(theta, 0) 
qc2.barrier()
qc2.h(1)         
qc2.cx(1, 2)
qc2.barrier()
qc2.cx(0, 1)     
qc2.h(0)
qc2.barrier()
qc2.measure([0, 1], c_alice_2) 
qc2.barrier()

# ... EXCEPTO AQUÍ: SE ELIMINAN LAS CORRECCIONES CLÁSICAS

# Verificación
qc2.ry(-theta, 2)
qc2.measure(2, c_bob_2)

# Ejecución en simulador ideal
sim2_sampler = AerSampler(run_options={"shots": SHOTS})
result2 = sim2_sampler.run([qc2]).result()
counts2 = result2.quasi_dists[0].binary_probabilities()

# Visualización de resultados
counts2_int = {k: int(v * SHOTS) for k, v in counts2.items()}
print(f"Resultados Probabilísticos: {counts2_int}")

fig2 = plot_histogram(counts2_int, color='darkorange', title='Simulación 2: Probabilística (1997)')
display(fig2)

# Visualización del circuito
print("Esquema del Circuito 2:")
display(qc2.draw('mpl'))

# %% [MD]
# --- 4. SIMULACIÓN 3: HARDWARE REAL ---
# Ejecución en la nube de IBM para observar el ruido.

# %%
print("\n--- Ejecutando Simulación 3: Hardware Real ---")

try:
    # 1. Buscar backend real menos ocupado
    backend_real = service.least_busy(simulator=False, operational=True, min_num_qubits=3)
    print(f"Backend elegido: {backend_real.name}")

    # 2. Transpilación del circuito ideal
    pm = generate_preset_pass_manager(backend=backend_real, optimization_level=1)
    tp_qc1 = pm.run(qc1)
    
    # Mostrar circuito transpilado
    print("Esquema del Circuito Transpilado (Adaptado al Hardware):")
    display(tp_qc1.draw('mpl', idle_wires=False, fold=20))

    # 3. Ejecución
    sampler_real = Sampler(mode=backend_real)
    print("Enviando trabajo a IBM (esto puede tardar unos minutos)...")
    job3 = sampler_real.run([tp_qc1], shots=SHOTS)
    print(f"ID del job: {job3.job_id()}")
    
    # Esperar resultados
    result3 = job3.result()
    print("Resultados recibidos.")

    # 4. Análisis
    pub_result = result3[0]
    counts3 = pub_result.data['bob_verif'].get_counts()
    
    fidelidad = counts3.get('0', 0) / SHOTS
    print(f"Fidelidad observada: {fidelidad*100:.2f}%")

    # Visualización
    fig3 = plot_histogram(counts3, color='crimson', title=f'Simulación 3: Hardware Real ({backend_real.name})')
    display(fig3)

except Exception as e:
    print(f"No se pudo ejecutar en hardware real. Error: {e}")

