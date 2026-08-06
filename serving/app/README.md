# 🎮 Aplicación Web - Predictor de Abandono de Usuarios

Una aplicación web moderna y responsive para predecir la probabilidad de abandono de usuarios del juego "Flood It" en tiempo real usando el modelo de Machine Learning desplegado con MLServer.

## ✨ Características

- **🎨 Diseño Moderno**: Interfaz atractiva con gradientes y animaciones CSS
- **📱 Responsive**: Funciona perfectamente en desktop, tablet y móvil  
- **⚡ Tiempo Real**: Predicciones instantáneas al enviar el formulario
- **📊 Visualización**: Barras de probabilidad animadas y resultados claros
- **🔍 Validación**: Validación de campos y manejo de errores
- **📝 Datos de Ejemplo**: Botón para llenar automáticamente datos de prueba

## 🏗️ Estructura

```
app/
├── index.html          # Aplicación web principal
└── README.md          # Esta documentación
```

## 📋 Campos del Formulario

### Información Básica
- **ID del Usuario**: Identificador único del usuario
- **Primera Interacción**: Fecha y hora de la primera sesión
- **Sistema Operativo**: Android o iOS
- **Idioma del Dispositivo**: Idioma configurado en el dispositivo
- **País**: País de origen del usuario

### Métricas de Juego
- **Interacciones del Usuario**: Número total de interacciones
- **Niveles Iniciados/Terminados/Completados**: Estadísticas de gameplay
- **Niveles Reiniciados**: Veces que el usuario reinició niveles
- **Puntuaciones Publicadas**: Scores compartidos
- **Gasto de Moneda Virtual**: Uso de moneda del juego
- **Recompensas por Anuncios**: Anuncios visionados
- **Desafíos a Amigos**: Invitaciones sociales enviadas
- **Completó 5 Niveles**: Logro de 5 niveles seguidos
- **Uso de Pasos Extra**: Funciones premium utilizadas

## 🚀 Cómo Usar

### 1. Iniciar el Servidor MLServer

Primero, asegúrate de que el servidor de ML esté ejecutándose:

```bash
cd ../serving
./start_server.sh
```

### 2. Abrir la Aplicación Web

Abre el archivo `index.html` en tu navegador web:

```bash
# Opción 1: Abrir directamente
open index.html

# Opción 2: Servidor local simple con Python
cd app
python -m http.server 8000
# Luego visita: http://localhost:8000

# Opción 3: Servidor local con Node.js
npx serve .
```

### 3. Usar la Aplicación

1. **Llenar el formulario** con los datos del usuario
2. **Usar "Llenar Datos de Ejemplo"** para pruebas rápidas
3. **Hacer clic en "Predecir Abandono"** para obtener resultados
4. **Ver la predicción** con probabilidades visuales

## 📊 Interpretación de Resultados

### Predicción Principal
- **✅ El usuario probablemente CONTINUARÁ**: Baja probabilidad de abandono
- **⚠️ El usuario probablemente ABANDONARÁ**: Alta probabilidad de abandono

### Barras de Probabilidad
- **🟢 No Abandonará**: Porcentaje de probabilidad de retención
- **🔴 Abandonará**: Porcentaje de probabilidad de abandono

### Recomendaciones Automáticas
La aplicación proporciona interpretaciones contextuales:
- **Alto riesgo (>80%)**: Intervención inmediata recomendada
- **Riesgo moderado (60-80%)**: Estrategias de retención
- **Riesgo bajo (<60%)**: Monitoreo continuo
- **Usuario comprometido (<20%)**: Candidato para programas premium

## 🛠️ Tecnologías Utilizadas

- **HTML5**: Estructura semántica moderna
- **CSS3**: Diseño con Flexbox, Grid y gradientes
- **JavaScript ES6+**: Funcionalidad asíncrona y fetch API
- **Responsive Design**: Mobile-first approach

## 🔧 Características Técnicas

### Validación de Datos
- Validación del lado cliente antes del envío
- Conversión automática de tipos de datos
- Manejo de campos requeridos

### Comunicación con la API
- Uso de fetch API para comunicación asíncrona
- Payload formateado según especificación MLServer
- Manejo robusto de errores HTTP

### UX/UI
- Loading states durante predicciones
- Animaciones CSS suaves
- Feedback visual inmediato
- Diseño intuitivo y accesible

## 🐛 Solución de Problemas

### Error de Conexión
```
❌ Error: Failed to fetch
```
**Solución**: Verificar que MLServer esté ejecutándose en `http://localhost:8080`

### Error 404 del Modelo
```
❌ Error HTTP: 404 - Not Found
```
**Solución**: Verificar que el modelo `flood-it-churned` esté cargado correctamente

### Error de CORS
```
❌ Error: CORS policy
```
**Solución**: Usar un servidor local para servir la aplicación web

## 📸 Capturas de Pantalla

### Vista Desktop
- Diseño de dos columnas con formulario y resultados
- Gradientes modernos y elementos visuales atractivos

### Vista Mobile
- Diseño de una columna responsive
- Formulario optimizado para pantallas táctiles

## 🔮 Mejoras Futuras

### Funcionalidades
- [ ] Historial de predicciones
- [ ] Exportación de resultados
- [ ] Comparación de múltiples usuarios
- [ ] Gráficos más avanzados
- [ ] Modo oscuro/claro

### Técnicas
- [ ] Progressive Web App (PWA)
- [ ] Caché de resultados
- [ ] Validación avanzada
- [ ] Soporte para más idiomas
- [ ] Integración con bases de datos

## 📝 Ejemplo de Uso

1. **Abrir la aplicación** en el navegador
2. **Hacer clic en "Llenar Datos de Ejemplo"**
3. **Modificar valores** según sea necesario
4. **Hacer clic en "Predecir Abandono"**
5. **Revisar resultados** y recomendaciones

## 🤝 Contribución

Para contribuir a esta aplicación:

1. Hacer fork del proyecto
2. Crear una rama para la nueva funcionalidad
3. Implementar cambios con pruebas
4. Enviar pull request con descripción detallada

## 📄 Licencia

Este proyecto es parte del taller de ML y está disponible para fines educativos.