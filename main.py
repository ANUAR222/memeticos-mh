import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import time

class RutaTuristica:
    def __init__(self, num_puntos: int = 20):
        """Inicializa puntos turísticos con coordenadas aleatorias y tiempos de visita"""
        self.num_puntos = num_puntos
        # Genera coordenadas aleatorias para puntos turísticos
        self.coordenadas = np.random.rand(num_puntos, 2) * 100
        # Genera tiempos de visita aleatorios entre 1 y 3 horas
        self.tiempos_visita = np.random.uniform(1, 3, num_puntos)
        
    def calcular_tiempo_total(self, ruta: List[int]) -> float:
        """Calcula tiempo total incluyendo viajes y visitas"""
        tiempo_total = 0
        for i in range(len(ruta)):
            # Añade tiempo de visita para el punto actual
            tiempo_total += self.tiempos_visita[ruta[i]]
            # Añade tiempo de viaje al siguiente punto
            if i < len(ruta) - 1:
                tiempo_total += self.calcular_distancia(ruta[i], ruta[i + 1])
        return tiempo_total

    def calcular_distancia(self, punto1: int, punto2: int) -> float:
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt(np.sum((self.coordenadas[punto1] - self.coordenadas[punto2]) ** 2))

class AlgoritmoMemetico:
    def __init__(self, 
                 ruta_turistica: RutaTuristica,
                 tamano_poblacion: int = 100,
                 generaciones: int = 1000,
                 tasa_mutacion: float = 0.1,
                 frecuencia_busqueda_local: int = 10,
                 subconjunto_busqueda_local: float = 0.2,
                 lamarckiano: bool = True):
        self.ruta_turistica = ruta_turistica
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
        self.frecuencia_busqueda_local = frecuencia_busqueda_local
        self.subconjunto_busqueda_local = subconjunto_busqueda_local
        self.lamarckiano = lamarckiano
        self.historial_mejor_aptitud = []
        self.historial_aptitud_promedio = []
        
    def inicializar_poblacion(self) -> List[List[int]]:
        """Crea población inicial de permutaciones aleatorias"""
        poblacion = []
        for _ in range(self.tamano_poblacion):
            ruta = list(range(self.ruta_turistica.num_puntos))
            random.shuffle(ruta)
            poblacion.append(ruta)
        return poblacion

    def aptitud(self, ruta: List[int]) -> float:
        """Calcula aptitud (inverso del tiempo total)"""
        tiempo_total = self.ruta_turistica.calcular_tiempo_total(ruta)
        return 1.0 / tiempo_total

    def seleccion_torneo(self, poblacion: List[List[int]], tamano_torneo: int = 3) -> List[int]:
        """Selecciona individuo usando selección por torneo"""
        torneo = random.sample(poblacion, tamano_torneo)
        return max(torneo, key=self.aptitud)

    def cruce_orden(self, padre1: List[int], padre2: List[int]) -> List[int]:
        """Implementa Cruce de Orden (OX) para permutación"""
        tamano = len(padre1)
        inicio, fin = sorted(random.sample(range(tamano), 2))
        
        # Inicializa descendiente con valores vacíos
        descendiente = [-1] * tamano
        
        # Copia segmento del padre1
        descendiente[inicio:fin] = padre1[inicio:fin]
        
        # Rellena posiciones restantes con elementos del padre2
        pos_actual = fin
        for elemento in padre2[fin:] + padre2[:fin]:
            if elemento not in descendiente:
                descendiente[pos_actual % tamano] = elemento
                pos_actual += 1
                
        return descendiente

    def mutacion_intercambio(self, ruta: List[int]) -> List[int]:
        """Aplica mutación de intercambio con probabilidad dada"""
        if random.random() < self.tasa_mutacion:
            i, j = random.sample(range(len(ruta)), 2)
            ruta[i], ruta[j] = ruta[j], ruta[i]
        return ruta

    def busqueda_local_2opt(self, ruta: List[int], max_iteraciones: int = 100) -> List[int]:
        """Aplica búsqueda local 2-opt para mejorar ruta"""
        mejor_ruta = ruta.copy()
        mejor_tiempo = self.ruta_turistica.calcular_tiempo_total(mejor_ruta)
        mejorado = True
        iteracion = 0
        
        while mejorado and iteracion < max_iteraciones:
            mejorado = False
            iteracion += 1
            
            for i in range(1, len(ruta) - 2):
                for j in range(i + 1, len(ruta)):
                    nueva_ruta = mejor_ruta.copy()
                    nueva_ruta[i:j] = nueva_ruta[i:j][::-1]  # Invierte segmento
                    nuevo_tiempo = self.ruta_turistica.calcular_tiempo_total(nueva_ruta)
                    
                    if nuevo_tiempo < mejor_tiempo:
                        mejor_ruta = nueva_ruta
                        mejor_tiempo = nuevo_tiempo
                        mejorado = True
                        break
                if mejorado:
                    break
                    
        return mejor_ruta

    def aplicar_busqueda_local(self, poblacion: List[List[int]], generacion: int) -> List[List[int]]:
        """Aplica búsqueda local según estrategia (estática o dinámica)"""
        if self.frecuencia_busqueda_local == 0:  # Estrategia estática
            if generacion == self.generaciones - 1:
                poblacion = [self.busqueda_local_2opt(ruta) for ruta in poblacion]
        else:  # Estrategia dinámica
            if generacion % self.frecuencia_busqueda_local == 0:
                num_individuos = int(self.tamano_poblacion * self.subconjunto_busqueda_local)
                indices = random.sample(range(self.tamano_poblacion), num_individuos)
                for idx in indices:
                    ruta_mejorada = self.busqueda_local_2opt(poblacion[idx])
                    if self.lamarckiano:
                        poblacion[idx] = ruta_mejorada
                    else:  # Modelo Baldwiniano
                        if self.aptitud(ruta_mejorada) > self.aptitud(poblacion[idx]):
                            poblacion[idx] = ruta_mejorada
        return poblacion

    def evolucionar(self) -> Tuple[List[int], List[float], List[float]]:
        """Ejecuta el algoritmo memético"""
        poblacion = self.inicializar_poblacion()
        mejor_solucion = None
        mejor_aptitud = float('-inf')
        
        for generacion in range(self.generaciones):
            # Aplica búsqueda local
            poblacion = self.aplicar_busqueda_local(poblacion, generacion)
            
            # Crea nueva población
            nueva_poblacion = []
            
            # Elitismo - mantiene mejor individuo
            elite = max(poblacion, key=self.aptitud)
            nueva_poblacion.append(elite)
            
            # Genera resto de nueva población
            while len(nueva_poblacion) < self.tamano_poblacion:
                padre1 = self.seleccion_torneo(poblacion)
                padre2 = self.seleccion_torneo(poblacion)
                descendiente = self.cruce_orden(padre1, padre2)
                descendiente = self.mutacion_intercambio(descendiente)
                nueva_poblacion.append(descendiente)
            
            poblacion = nueva_poblacion
            
            # Registra estadísticas
            aptitudes_actuales = [self.aptitud(ruta) for ruta in poblacion]
            mejor_actual = max(aptitudes_actuales)
            promedio_actual = sum(aptitudes_actuales) / len(aptitudes_actuales)
            
            self.historial_mejor_aptitud.append(mejor_actual)
            self.historial_aptitud_promedio.append(promedio_actual)
            
            if mejor_actual > mejor_aptitud:
                mejor_aptitud = mejor_actual
                mejor_solucion = poblacion[aptitudes_actuales.index(mejor_actual)]
                
        return mejor_solucion, self.historial_mejor_aptitud, self.historial_aptitud_promedio

def comparar_algoritmos():
    """Compara diferentes variantes del algoritmo"""
    np.random.seed(42)
    random.seed(42)
    
    # Instancia del problema
    problema_turistico = RutaTuristica(num_puntos=20)
    
    # Variantes del algoritmo
    variantes = [
        ("Algoritmo Genético", {"frecuencia_busqueda_local": 0}),
        ("Memético Estático (Lamarckiano)", {"frecuencia_busqueda_local": 0, "lamarckiano": True}),
        ("Memético Dinámico (Baldwiniano)", {"frecuencia_busqueda_local": 10, "lamarckiano": False})
    ]
    
    resultados = {}
    
    for nombre, params in variantes:
        print(f"\nEjecutando {nombre}...")
        tiempo_inicio = time.time()
        
        algoritmo = AlgoritmoMemetico(problema_turistico, **params)
        mejor_ruta, historial_mejor, historial_promedio = algoritmo.evolucionar()
        
        tiempo_ejecucion = time.time() - tiempo_inicio
        tiempo_final = problema_turistico.calcular_tiempo_total(mejor_ruta)
        
        resultados[nombre] = {
            "mejor_ruta": mejor_ruta,
            "tiempo_final": tiempo_final,
            "tiempo_ejecucion": tiempo_ejecucion,
            "historial_mejor": historial_mejor,
            "historial_promedio": historial_promedio
        }
        
        print(f"Tiempo final de ruta: {tiempo_final:.2f}")
        print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
    
    # Grafica comparación de convergencia
    plt.figure(figsize=(12, 6))
    for nombre, datos in resultados.items():
        plt.plot(datos["historial_mejor"], label=f"{nombre} (Mejor)")
    plt.xlabel("Generación")
    plt.ylabel("Aptitud")
    plt.title("Comparación de Convergencia")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparacion_convergencia.png")
    plt.close()
    
    return resultados

if __name__ == "__main__":
    resultados = comparar_algoritmos()
