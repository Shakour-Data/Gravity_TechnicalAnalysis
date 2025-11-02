# 📚 Runbook - راهنمای عملیاتی Technical Analysis Microservice

## 🎯 هدف
این Runbook راهنمای جامعی برای نگهداری، عیب‌یابی و مدیریت عملیاتی میکروسرویس Technical Analysis است.

---

## 📋 فهرست مطالب
1. [معماری و اجزا](#معماری-و-اجزا)
2. [راه‌اندازی و Deployment](#راه‌اندازی-و-deployment)
3. [Monitoring و Alerts](#monitoring-و-alerts)
4. [Troubleshooting](#troubleshooting)
5. [Backup و Recovery](#backup-و-recovery)
6. [Scaling Strategies](#scaling-strategies)
7. [Security Procedures](#security-procedures)

---

## 🏗️ معماری و اجزا

### نمای کلی
```
┌─────────────────────────────────────────────────────┐
│  Load Balancer / Ingress                            │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │   API Gateway  │
       └───────┬────────┘
               │
    ┌──────────▼───────────┐
    │ Technical Analysis   │
    │   Microservice       │
    │  (3-20 replicas)     │
    └──────────┬───────────┘
               │
    ┌──────────┴───────────┐
    │                      │
┌───▼────┐          ┌─────▼─────┐
│ Redis  │          │  Kafka /  │
│ Cache  │          │ RabbitMQ  │
└────────┘          └───────────┘
```

### اجزای اصلی
1. **API Layer**: FastAPI با 3+ replicas
2. **Cache Layer**: Redis cluster
3. **Message Queue**: Kafka/RabbitMQ (اختیاری)
4. **Service Discovery**: Eureka/Consul
5. **Observability**: Prometheus + Jaeger

---

## 🚀 راه‌اندازی و Deployment

### Prerequisites
- Kubernetes cluster (v1.25+)
- kubectl configured
- Helm 3.x (اختیاری)
- Docker registry access

### Deployment Steps

#### 1. تنظیم Namespace
```bash
kubectl apply -f k8s/namespace.yaml
```

#### 2. تنظیم Secrets
```bash
# ایجاد secret برای credentials
kubectl create secret generic technical-analysis-secret \
  --from-literal=SECRET_KEY="your-secret-key" \
  --from-literal=REDIS_PASSWORD="redis-password" \
  -n tech-analysis-prod

# برای استفاده از Vault (توصیه می‌شود):
kubectl apply -f k8s/vault-secret-sync.yaml
```

#### 3. Deploy Redis
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis \
  --namespace tech-analysis-prod \
  --set auth.password=secure-password \
  --set master.persistence.size=10Gi \
  --set replica.replicaCount=2
```

#### 4. Deploy Application
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

#### 5. بررسی وضعیت
```bash
# وضعیت pods
kubectl get pods -n tech-analysis-prod

# logs
kubectl logs -f deployment/technical-analysis -n tech-analysis-prod

# health check
kubectl run test --rm -i --restart=Never \
  --image=curlimages/curl \
  -- curl http://technical-analysis-service:8000/health
```

### Rolling Update
```bash
# به‌روزرسانی image
kubectl set image deployment/technical-analysis \
  technical-analysis=ghcr.io/gravitywavesml/gravity_techanalysis:v1.1.0 \
  -n tech-analysis-prod

# نظارت بر rollout
kubectl rollout status deployment/technical-analysis -n tech-analysis-prod

# در صورت مشکل، rollback
kubectl rollout undo deployment/technical-analysis -n tech-analysis-prod
```

### Canary Deployment
```bash
# ایجاد نسخه canary
kubectl apply -f k8s/deployment-canary.yaml

# تنظیم traffic split (20% canary)
kubectl apply -f k8s/virtual-service-canary.yaml

# مانیتور metrics برای 30 دقیقه
# اگر موفق: promote canary
kubectl apply -f k8s/deployment-canary-full.yaml

# اگر ناموفق: rollback
kubectl delete -f k8s/deployment-canary.yaml
```

---

## 📊 Monitoring و Alerts

### Key Metrics

#### Application Metrics
```prometheus
# Request Rate
rate(http_requests_total[5m])

# Error Rate
rate(http_requests_total{status=~"5.."}[5m])

# Response Time (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Active Requests
http_requests_active

# Cache Hit Rate
rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])
```

#### System Metrics
```prometheus
# CPU Usage
container_cpu_usage_seconds_total

# Memory Usage
container_memory_usage_bytes

# Pod Restart Count
kube_pod_container_status_restarts_total

# Replica Count
kube_deployment_status_replicas_available
```

### Alert Rules

#### Critical Alerts (P1)
```yaml
# High Error Rate
alert: HighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
for: 2m
severity: critical
action: "Page on-call engineer"

# Service Down
alert: ServiceDown
expr: up{job="technical-analysis"} == 0
for: 1m
severity: critical
action: "Immediate investigation"

# Pod Crash Loop
alert: PodCrashLooping
expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
severity: critical
action: "Check pod logs"
```

#### Warning Alerts (P2)
```yaml
# High Response Time
alert: HighResponseTime
expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
for: 5m
severity: warning
action: "Monitor for scaling"

# Redis Connection Issues
alert: RedisCacheDown
expr: redis_up == 0
for: 2m
severity: warning
action: "Check Redis cluster health"

# Low Cache Hit Rate
alert: LowCacheHitRate
expr: rate(cache_hits_total[5m]) / rate(cache_requests_total[5m]) < 0.5
for: 10m
severity: warning
action: "Review cache configuration"
```

### Dashboards

#### Grafana Dashboard IDs
- **Overview**: `tech-analysis-overview`
- **Performance**: `tech-analysis-performance`
- **Business Metrics**: `tech-analysis-business`
- **Infrastructure**: `tech-analysis-infrastructure`

#### Key Panels
1. Request Rate (requests/sec)
2. Error Rate (%)
3. Response Time (p50, p95, p99)
4. CPU & Memory Usage
5. Cache Hit Rate
6. Active Connections
7. Top Endpoints
8. Error Breakdown by Type

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Service Not Responding (503)**

**علائم:**
- Health checks failing
- 503 errors من load balancer
- Pods در حالت `CrashLoopBackOff`

**تشخیص:**
```bash
# بررسی وضعیت pods
kubectl get pods -n tech-analysis-prod

# بررسی logs
kubectl logs -f pod/technical-analysis-xxx -n tech-analysis-prod

# بررسی events
kubectl describe pod/technical-analysis-xxx -n tech-analysis-prod

# بررسی health endpoint
kubectl port-forward svc/technical-analysis-service 8000:8000
curl http://localhost:8000/health/ready
```

**راه‌حل:**
```bash
# اگر Redis down است:
kubectl get pods -l app=redis -n tech-analysis-prod
kubectl logs -f redis-master-0 -n tech-analysis-prod

# اگر مشکل configuration است:
kubectl get configmap technical-analysis-config -o yaml
kubectl edit configmap technical-analysis-config

# restart pods
kubectl rollout restart deployment/technical-analysis -n tech-analysis-prod
```

---

#### 2. **High Response Time (> 2s)**

**علائم:**
- p95 latency > 2 seconds
- Client timeouts
- Queue buildup

**تشخیص:**
```bash
# بررسی CPU/Memory
kubectl top pods -n tech-analysis-prod

# بررسی HPA
kubectl get hpa -n tech-analysis-prod

# بررسی slow queries در logs
kubectl logs -f deployment/technical-analysis \
  -n tech-analysis-prod | grep "duration"
```

**راه‌حل:**
```bash
# افزایش replicas (اگر CPU بالاست)
kubectl scale deployment/technical-analysis --replicas=10 -n tech-analysis-prod

# بررسی cache
redis-cli info stats | grep keyspace_hits

# اگر cache miss rate بالاست:
# افزایش TTL یا review کردن cache strategy

# اگر database slow است:
# بررسی indexes و queries
```

---

#### 3. **Memory Leak / OOM Kills**

**علائم:**
- Pods restart frequently
- OOMKilled status
- Memory usage رو به افزایش

**تشخیص:**
```bash
# بررسی memory usage history
kubectl top pods -n tech-analysis-prod --watch

# بررسی OOM kills
kubectl get events -n tech-analysis-prod | grep OOM

# پروفایل memory
kubectl exec -it technical-analysis-xxx -n tech-analysis-prod -- sh
# در container:
pip install memory_profiler
python -m memory_profiler main.py
```

**راه‌حل:**
```bash
# افزایش memory limits (موقت)
kubectl edit deployment technical-analysis -n tech-analysis-prod
# resources.limits.memory: 4Gi

# بررسی کد برای leaks:
# - Connection pools not closed
# - Large objects in memory
# - Circular references

# Redeploy با fix
```

---

#### 4. **Redis Connection Errors**

**علائم:**
- `ConnectionError: Error connecting to Redis`
- Cache misses زیاد
- Timeouts

**تشخیص:**
```bash
# بررسی Redis health
kubectl exec -it redis-master-0 -n tech-analysis-prod -- redis-cli ping

# بررسی connections
kubectl exec -it redis-master-0 -n tech-analysis-prod -- redis-cli info clients

# بررسی network
kubectl exec -it technical-analysis-xxx -n tech-analysis-prod -- \
  telnet redis-service 6379
```

**راه‌حل:**
```bash
# اگر Redis down است:
kubectl rollout restart statefulset redis -n tech-analysis-prod

# اگر connection pool exhausted است:
# افزایش max_connections در config

# اگر network issue است:
kubectl get networkpolicies -n tech-analysis-prod
```

---

#### 5. **High Error Rate (5xx)**

**علائم:**
- 500/503 errors spike
- Alert fired
- User complaints

**تشخیص:**
```bash
# بررسی error logs
kubectl logs -f deployment/technical-analysis \
  -n tech-analysis-prod | grep ERROR

# بررسی error breakdown
# در Grafana یا Kibana

# trace errors
# در Jaeger UI
```

**راه‌حل:**
```bash
# اگر dependency down است:
kubectl get svc -n tech-analysis-prod

# اگر bug در code است:
# Hotfix & redeploy

# اگر rate limit است:
# افزایش limits یا scaling

# در ضرورت: rollback
kubectl rollout undo deployment/technical-analysis -n tech-analysis-prod
```

---

## 💾 Backup و Recovery

### Backup Strategy

#### 1. Configuration Backup
```bash
# Daily backup همه configs
kubectl get all,configmap,secret -n tech-analysis-prod -o yaml > backup-$(date +%Y%m%d).yaml

# Automated backup
# در CronJob
```

#### 2. Redis Backup
```bash
# Manual backup
kubectl exec redis-master-0 -n tech-analysis-prod -- redis-cli SAVE

# Copy RDB file
kubectl cp redis-master-0:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb

# Automated: استفاده از Redis backup tool یا Velero
```

### Recovery Procedures

#### Disaster Recovery
```bash
# 1. بازگرداندن از backup
kubectl apply -f backup-20240101.yaml

# 2. Restore Redis
kubectl cp redis-backup.rdb redis-master-0:/data/dump.rdb
kubectl exec redis-master-0 -- redis-cli SHUTDOWN
# Redis auto-restart و load می‌کند

# 3. بررسی health
kubectl get pods -n tech-analysis-prod
```

---

## 📈 Scaling Strategies

### Horizontal Scaling (HPA)
```yaml
# Auto-scaling based on CPU & Memory
Current: 3-20 replicas
Triggers:
  - CPU > 70%: scale up
  - Memory > 80%: scale up
  - Request rate > 1000/s: scale up
```

### Manual Scaling
```bash
# Scale up
kubectl scale deployment/technical-analysis --replicas=15 -n tech-analysis-prod

# Scale down (off-peak hours)
kubectl scale deployment/technical-analysis --replicas=5 -n tech-analysis-prod
```

### Vertical Scaling
```bash
# افزایش resources
kubectl edit deployment technical-analysis -n tech-analysis-prod

# Update:
resources:
  requests:
    cpu: 1000m
    memory: 1Gi
  limits:
    cpu: 4000m
    memory: 4Gi
```

---

## 🔒 Security Procedures

### Security Checklist
- [ ] تمام secrets در Vault
- [ ] TLS برای تمام connections
- [ ] Network policies فعال
- [ ] RBAC به درستی تنظیم شده
- [ ] Security headers فعال
- [ ] Rate limiting فعال
- [ ] Input validation فعال
- [ ] Audit logging فعال

### Incident Response
1. **تشخیص**: Alerts → PagerDuty
2. **تحلیل**: Logs + Traces
3. **Contain**: Isolate affected pods
4. **Recover**: Redeploy/Rollback
5. **Post-mortem**: Document lessons

### Security Updates
```bash
# بررسی vulnerabilities
trivy image ghcr.io/gravitywavesml/gravity_techanalysis:latest

# Update dependencies
pip list --outdated
pip install --upgrade <package>

# Redeploy
kubectl rollout restart deployment/technical-analysis -n tech-analysis-prod
```

---

## 📞 On-Call Contact

| Role | Contact | Escalation |
|------|---------|------------|
| Primary On-Call | +1-xxx-xxx-xxxx | 5 min |
| Secondary On-Call | +1-xxx-xxx-xxxx | 15 min |
| Team Lead | +1-xxx-xxx-xxxx | 30 min |
| Director | +1-xxx-xxx-xxxx | 1 hour |

---

## 📚 مراجع

- **API Documentation**: https://api.example.com/docs
- **Grafana**: https://grafana.example.com
- **Jaeger**: https://jaeger.example.com
- **Kubernetes Dashboard**: https://k8s.example.com
- **Runbook Updates**: [GitHub Wiki](https://github.com/...)
