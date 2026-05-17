# Runbook: Resolver consumer lag

## Cuándo usarlo
- El dashboard de Grafana muestra acumulación de mensajes sin procesar.
- El consumer o inference_consumer está `unhealthy` o con lag creciente.
- El sistema de feature engineering no está actualizando TimescaleDB.

## Diagnóstico

### Opción A — Kafka UI (visual)
Abre `http://localhost:8080` → Consumer Groups → busca `fraud-feature-engineering`.
El campo **Lag** por partición indica mensajes pendientes.

### Opción B — CLI desde el contenedor de Kafka

```bash
# Consumer group del consumer (feature engineering):
docker compose exec -T kafka kafka-consumer-groups \
  --bootstrap-server localhost:29092 \
  --group fraud-feature-engineering \
  --describe

# Ver el estado del consumer y del inference_consumer:
docker compose ps consumer inference_consumer
docker compose logs --tail=50 consumer
docker compose logs --tail=50 inference_consumer
```

## Causas frecuentes y soluciones

### 1. Contenedor caído o reiniciándose

```bash
docker compose ps consumer
# Si aparece "Restarting" o "Exited":
docker compose logs --tail=100 consumer
docker compose up -d --force-recreate consumer
```

### 2. Redis no disponible (consumer en modo degradado)

El consumer tiene circuit breaker para Redis. Si Redis falla, continúa procesando sin caché pero con mayor latencia.

```bash
docker compose ps redis
docker compose logs --tail=50 redis
docker compose restart redis
# Redis se recupera en ~5s; el consumer reconecta automáticamente.
```

### 3. TimescaleDB saturado

```bash
docker compose exec -T timescaledb psql -U "${TIMESCALE_USER}" -d "${TIMESCALE_DB}" -c \
  "SELECT COUNT(*) FROM public.transactions WHERE timestamp > NOW() - INTERVAL '5 minutes';"
# Si el count no crece, el consumer no está escribiendo.

# Verificar conexiones activas:
docker compose exec -T timescaledb psql -U "${TIMESCALE_USER}" -d "${TIMESCALE_DB}" -c \
  "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
```

### 4. Lag estructural (producer más rápido que consumer)

Si el producer genera más mensajes de los que el consumer puede procesar:

```bash
# Opción temporal: aumentar el número de instancias del consumer:
docker compose up -d --scale consumer=2
# Nota: con más de 3 particiones en transactions.raw, pueden correr hasta 3 instancias en paralelo.
```

## Resetear offsets (solo en caso de corrupción)

⚠️ **Peligroso**: resetear offsets hace que el consumer reprocese mensajes. Puede generar duplicados en TimescaleDB (mitigados por el `ON CONFLICT DO NOTHING`).

```bash
# Solo ejecutar si se confirma corrupción de offsets:
docker compose stop consumer
docker compose exec -T kafka kafka-consumer-groups \
  --bootstrap-server localhost:29092 \
  --group fraud-feature-engineering \
  --topic transactions.raw \
  --reset-offsets --to-latest --execute
docker compose up -d consumer
```

## Verificación post-resolución

```bash
# El lag debe bajar a 0 en los segundos siguientes al reinicio:
docker compose exec -T kafka kafka-consumer-groups \
  --bootstrap-server localhost:29092 \
  --group fraud-feature-engineering \
  --describe
# CONSUMER-ID debe estar poblado y LAG debe decrecer o ser 0.
```
