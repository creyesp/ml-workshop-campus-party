# Despliegue MLServer - Guía de Inicio

Este directorio contiene la configuración de despliegue de MLServer para el modelo de predicción flood-it-churned con un manejador personalizado que acepta entrada JSON.

## 📋 Requisitos Previos

- Python 3.8+ con entorno virtual activado
- MLServer instalado (`pip install mlserver mlserver-sklearn`)
- Archivo de modelo entrenado: `../notebooks/models/xgb_model_full.joblib`

## 🚀 Inicio Rápido

### Opción 1: Usando Scripts de Conveniencia (Recomendado)

```bash
# Iniciar el servidor
./start_server.sh

# En otra terminal, probar el despliegue
./test_deployment.sh
```

### Opción 2: Inicio Manual

```bash
# Desde el directorio serving/
cd /ruta/a/ml-workshop-campus-party/serving
mlserver start .
```

El servidor se iniciará en:
- **API HTTP**: http://localhost:8080
- **API gRPC**: http://localhost:8081  
- **Métricas**: http://localhost:8082

### 2. Verificar Estado del Servidor

```bash
# Verificar si el modelo está cargado
curl http://localhost:8080/v2/models/flood-it-churned
```

Respuesta esperada:
```json
{
  "name": "flood-it-churned",
  "versions": [],
  "platform": "",
  "inputs": [],
  "outputs": [],
  "parameters": {}
}
```

### 3. Probar Predicción con Datos JSON

```bash
curl -X POST http://localhost:8080/v2/models/flood-it-churned/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [{
      "name": "json_data",
      "shape": [1],
      "datatype": "BYTES",
      "data": [{
        "user_first_engagement": "2018-07-01 23:26:11.222000+00:00",
        "user_pseudo_id": "ABC123",
        "is_enable": 1,
        "bounced": 0,
        "country_name": "United States",
        "device_os": "IOS",
        "device_lang": "en",
        "cnt_user_engagement": 5,
        "cnt_level_start_quickplay": 2,
        "cnt_level_end_quickplay": 1,
        "cnt_level_complete_quickplay": 1,
        "cnt_level_reset_quickplay": 0,
        "cnt_post_score": 2,
        "cnt_spend_virtual_currency": 0,
        "cnt_ad_reward": 1,
        "cnt_challenge_a_friend": 0,
        "cnt_completed_5_levels": 0,
        "cnt_use_extra_steps": 0
      }]
    }]
  }'
```

## 📁 Descripción de Archivos

- **`start_server.sh`** - Script de conveniencia para iniciar servidor con verificaciones de salud
- **`test_deployment.sh`** - Script de prueba rápida para verificar despliegue
- **`model-settings.json`** - Configuración del modelo usando manejador personalizado
- **`settings.json`** - Configuración de MLServer (puertos, workers, etc.)
- **`custom_sklearn_handler.py`** - Manejador personalizado para conversión JSON→DataFrame
- **`start_server.sh`** - Script de conveniencia para iniciar servidor
- **`mlserver.log`** - Logs del servidor (creado al ejecutar)

## 🔧 Detalles de Configuración

### Configuración del Modelo (`model-settings.json`)
```json
{
    "name": "flood-it-churned",
    "implementation": "custom_sklearn_handler.CustomSklearnModel",
    "parameters": {
        "uri": "/ruta/absoluta/a/xgb_model_full.joblib",
        "version": "v0.1.0"
    }
}
```

### Configuración del Servidor (`settings.json`)
```json
{
    "debug": true,
    "parallel_workers": 0,
    "http_port": 8080,
    "grpc_port": 8081,
    "metrics_port": 8082,
    "use_single_worker": true
}
```

## 🧪 Scripts de Prueba

### Script de Prueba en Python
```bash
# Ejecutar prueba integral desde la raíz del proyecto
cd /ruta/a/ml-workshop-campus-party
python notebooks/test_custom_handler.py
```

Este script hará:
- ✅ Verificar estado del servidor
- ✅ Enviar datos JSON de muestra
- ✅ Mostrar predicciones y probabilidades
- ✅ Mostrar comandos curl de ejemplo

## 📊 Formato de Datos de Entrada

El manejador personalizado acepta objetos JSON con estos campos requeridos:

```json
{
  "user_first_engagement": "2018-07-01 23:26:11.222000+00:00",
  "user_pseudo_id": "id_usuario_unico",
  "is_enable": 1,
  "bounced": 0,
  "country_name": "United States",
  "device_os": "IOS",
  "device_lang": "en",
  "cnt_user_engagement": 5,
  "cnt_level_start_quickplay": 2,
  "cnt_level_end_quickplay": 1,
  "cnt_level_complete_quickplay": 1,
  "cnt_level_reset_quickplay": 0,
  "cnt_post_score": 2,
  "cnt_spend_virtual_currency": 0,
  "cnt_ad_reward": 1,
  "cnt_challenge_a_friend": 0,
  "cnt_completed_5_levels": 0,
  "cnt_use_extra_steps": 0
}
```

## 📈 Formato de Respuesta

El servidor devuelve predicciones y probabilidades:

```json
{
  "outputs": [
    {
      "name": "predictions",
      "shape": [1],
      "datatype": "INT64",
      "data": [0]  // 0 = No Abandonará, 1 = Abandonará
    },
    {
      "name": "probabilities",
      "shape": [1, 2], 
      "datatype": "FP64",
      "data": [0.825, 0.175]  // [prob_no_abandono, prob_abandono]
    }
  ]
}
```

## 🐛 Solución de Problemas

### El Servidor No Inicia
```bash
# Verificar si el puerto ya está en uso
lsof -i :8080

# Matar procesos MLServer existentes
pkill -f mlserver

# Verificar que el archivo del modelo existe
ls -la ../notebooks/models/xgb_model_full.joblib
```

### Errores de Carga del Modelo
- Verificar la ruta del archivo del modelo en `model-settings.json`
- Verificar que el modelo fue entrenado con una versión compatible de sklearn
- Asegurar que el archivo del manejador personalizado esté en el mismo directorio

### Errores de Predicción
- Verificar que todos los campos requeridos estén presentes en el JSON
- Verificar que los nombres de campos coincidan exactamente (sensible a mayúsculas)
- Revisar logs del servidor: `tail -f mlserver.log`

## 🌐 Ejemplos de Integración de Cliente

### Cliente Python
```python
import requests

def predecir_abandono(datos_usuario):
    payload = {
        "inputs": [{
            "name": "json_data",
            "shape": [1],
            "datatype": "BYTES",
            "data": [datos_usuario]
        }]
    }
    
    response = requests.post(
        'http://localhost:8080/v2/models/flood-it-churned/infer',
        json=payload
    )
    
    result = response.json()
    prediccion = result['outputs'][0]['data'][0]
    probabilidades = result['outputs'][1]['data']
    
    return {
        'abandonara': bool(prediccion),
        'probabilidad_abandono': probabilidades[1],
        'probabilidad_no_abandono': probabilidades[0]
    }
```

### Cliente JavaScript/Node.js
```javascript
async function predecirAbandono(datosUsuario) {
    const payload = {
        inputs: [{
            name: "json_data",
            shape: [1],
            datatype: "BYTES",
            data: [datosUsuario]
        }]
    };
    
    const response = await fetch('http://localhost:8080/v2/models/flood-it-churned/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    return {
        abandonara: Boolean(result.outputs[0].data[0]),
        probabilidadAbandono: result.outputs[1].data[1],
        probabilidadNoAbandono: result.outputs[1].data[0]
    };
}
```

## 📚 Recursos Adicionales

- [Documentación de MLServer](https://mlserver.readthedocs.io/)
- [Guía de Implementación de Modelo Personalizado](https://mlserver.readthedocs.io/en/latest/examples/custom/README.html)
- [Referencia de Configuración de Modelo](https://mlserver.readthedocs.io/en/latest/reference/model-settings.html)

## 🔄 Flujo de Trabajo de Desarrollo

1. **Hacer cambios** al manejador personalizado o configuración
2. **Reiniciar servidor**: `pkill -f mlserver && mlserver start .`
3. **Probar cambios**: `python ../notebooks/test_custom_handler.py`
4. **Verificar logs**: `tail -f mlserver.log`

## 🚀 Despliegue en Producción

Para producción, considerar:
- Usar contenedores Docker
- Agregar autenticación/autorización
- Configurar balanceo de carga
- Implementar monitoreo y alertas
- Usar endpoints HTTPS
- Configurar niveles de logging apropiados