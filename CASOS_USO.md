# Casos de Uso Prácticos - Detección y Conteo de Vehículos

## 🚗 Estacionamiento
- **Orientación**: horizontal (entrada lateral)
- **Posición línea**: 30% (cerca de la entrada)
- **Capacidad carros**: 100 (plazas disponibles)
- **Capacidad motos**: 20 (área de motos)
- **Conf**: 0.4 (balance para detectar bien en exterior)

## 🚦 Control de tráfico en calle
- **Orientación**: vertical (cámara cenital)
- **Posición línea**: 50% (centro de la calle)
- **Invertir dirección**: Según sentido del tráfico
- **Capacidades**: 999 (solo para estadísticas, no límites)
- **Conf**: 0.3 (detectar todo el tráfico posible)

## 🏢 Entrada de edificio/garage
- **Orientación**: horizontal
- **Posición línea**: 20% (justo después de la barrera)
- **Capacidad**: Según plazas disponibles
- **Conf**: 0.5 (alta precisión para evitar errores)

## 🛣️ Peaje automático
- **Orientación**: horizontal
- **Posición línea**: 75% (después del punto de pago)
- **Invertir dirección**: No
- **Capacidades**: 999 (solo conteo)
- **Conf**: 0.6 (alta precisión)

## 🏪 Drive-through
- **Orientación**: vertical
- **Posición línea**: Variable según punto de interés
  - 20% para entrada
  - 50% para ventanilla de pedido
  - 80% para ventanilla de entrega
- **Conf**: 0.4
- **Capacidades**: Según espacio de espera

## Tips para ajustar la posición de línea:

### Entradas:
- Usa **10-30%** para detectar vehículos justo al entrar
- Evita ponerla muy cerca del borde (menos del 10%)

### Salidas:
- Usa **70-90%** para detectar vehículos al salir
- Útil para confirmar que el vehículo completó el recorrido

### Zonas intermedias:
- Usa **40-60%** para análisis de flujo en medio del recorrido
- Ideal para detectar congestión o velocidad de paso

### Múltiples carriles:
- Ajusta la posición para enfocarte en un carril específico
- Por ejemplo, 25% para carril izquierdo, 75% para carril derecho

### Evitar obstáculos:
- Si hay árboles, postes o sombras en el centro, mueve la línea
- Busca la zona más despejada del video