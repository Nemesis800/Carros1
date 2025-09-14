# 🤝 Guía de Contribución - ConteoVePa

¡Gracias por tu interés en contribuir al proyecto de **Detección y Conteo de Vehículos**! Esta guía te ayudará a configurar el entorno y seguir las mejores prácticas para el desarrollo colaborativo.

## 🚀 Configuración Inicial

### 1. Clonar el repositorio
```bash
git clone https://github.com/Nemesis800/ConteoVePa.git
cd ConteoVePa
```

### 2. Configurar entorno virtual
```bash
# Usando UV (recomendado)
uv venv .venv
uv pip install -r requirements.txt

# O usando pip tradicional
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verificar instalación
```bash
uv run -p .venv python src/app.py
```

## 🔄 Flujo de Trabajo Colaborativo

### Antes de empezar a trabajar:
1. **Actualizar tu rama local:**
   ```bash
   git pull origin main
   ```

2. **Crear una nueva rama para tu característica:**
   ```bash
   git checkout -b feature/nombre-de-tu-caracteristica
   # Ejemplo: git checkout -b feature/mejora-deteccion-motos
   ```

### Durante el desarrollo:
3. **Hacer commits frecuentes y descriptivos:**
   ```bash
   git add .
   git commit -m "✨ Descripción clara del cambio realizado"
   ```

4. **Actualizar periódicamente desde main:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Al completar tu trabajo:
5. **Push de tu rama:**
   ```bash
   git push origin feature/nombre-de-tu-caracteristica
   ```

6. **Crear Pull Request:**
   - Ve a https://github.com/Nemesis800/ConteoVePa
   - Haz clic en "New Pull Request"
   - Describe qué cambios realizaste
   - Espera la revisión del equipo

## 📝 Convenciones de Código

### Estilo Python:
- Usar **PEP 8** como estándar
- Nombres de variables en `snake_case`
- Nombres de clases en `PascalCase`
- Docstrings para funciones y clases importantes

### Mensajes de Commit:
Use estos prefijos para categorizar tus commits:
- `✨ feature:` Nueva funcionalidad
- `🐛 fix:` Corrección de bugs
- `📚 docs:` Actualizaciones de documentación
- `🎨 style:` Cambios de formato/estilo
- `♻️ refactor:` Refactorización de código
- `🧪 test:` Añadir o modificar tests
- `🔧 config:` Cambios de configuración

## 🧪 Testing

### Antes de enviar tu Pull Request:
1. **Probar la aplicación:**
   ```bash
   uv run -p .venv python src/app.py
   ```

2. **Verificar con diferentes videos:**
   - Probar con webcam
   - Probar con archivos de video
   - Verificar configuraciones de línea

3. **Comprobar que no hay errores:**
   - La aplicación debe iniciar sin errores
   - El botón 'q' debe funcionar correctamente
   - La detención y reinicio debe ser fluida

## 🐛 Reportar Issues

Cuando encuentres un bug o tengas una sugerencia:

1. **Busca si ya existe un issue similar**
2. **Crea un nuevo issue con:**
   - Título descriptivo
   - Pasos para reproducir el problema
   - Comportamiento esperado vs actual
   - Screenshots si es aplicable
   - Tu sistema operativo y versión de Python

## 🎯 Áreas de Contribución

### Funcionalidades deseadas:
- 🎥 Soporte para más formatos de video
- 📊 Exportación de estadísticas
- 🎨 Mejoras en la interfaz gráfica
- 🚗 Detección de más tipos de vehículos
- 📱 Versión web o móvil
- 🔄 Sistema de configuración avanzada

### Mejoras técnicas:
- 🧪 Añadir tests automatizados
- 📈 Optimización de rendimiento
- 🛡️ Manejo de errores mejorado
- 📚 Documentación más detallada
- 🔒 Validación de datos de entrada

## 💬 Comunicación

### Discord/Teams del equipo:
- Canal principal: **#conteo-vepa-dev**
- Canal bugs: **#conteo-vepa-bugs**
- Canal ideas: **#conteo-vepa-ideas**

### Reuniones:
- **Stand-up diario**: 9:00 AM
- **Review semanal**: Viernes 3:00 PM
- **Sprint planning**: Cada 2 semanas

## 🏆 Reconocimientos

Todos los colaboradores serán reconocidos en:
- README principal del proyecto
- Sección de contributors en GitHub
- Documentación final del proyecto

## ❓ ¿Dudas?

Si tienes preguntas:
1. Busca en los **issues cerrados**
2. Pregunta en el **canal de Discord/Teams**
3. Crea un **issue** con la etiqueta `question`
4. Contacta directamente a @Nemesis800

---
## ⚡ Comandos Útiles de Referencia Rápida

```bash
# Clonar y configurar
git clone https://github.com/Nemesis800/ConteoVePa.git
cd ConteoVePa
uv venv .venv && uv pip install -r requirements.txt

# Crear nueva rama
git checkout -b feature/mi-nueva-caracteristica

# Trabajo diario
git pull origin main
# ... hacer cambios ...
git add .
git commit -m "✨ feature: descripción del cambio"
git push origin feature/mi-nueva-caracteristica

# Actualizar desde main
git fetch origin && git rebase origin/main
```

¡Gracias por contribuir al proyecto! 🚀