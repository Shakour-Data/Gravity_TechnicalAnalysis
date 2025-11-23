# Gravity Technical Analysis - Production Deployment Guide

**Version:** 1.1.0  
**Date:** November 12, 2025  
**Team:** Lars Andersson (DevOps Lead)  
**Status:** Production-Ready ✅

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Steps](#deployment-steps)
4. [Monitoring & Observability](#monitoring--observability)
5. [Scaling & Performance](#scaling--performance)
6. [Troubleshooting](#troubleshooting)
7. [Disaster Recovery](#disaster-recovery)
8. [Security](#security)

---

## 🎯 Prerequisites

### Required Tools
```bash
# Kubernetes CLI
kubectl version --client

# Helm 3
helm version

# Docker
docker --version

# Optional: k9s for cluster management
k9s version
```

### Kubernetes Cluster Requirements
- **Version:** 1.24+
- **Nodes:** Minimum 3 worker nodes
- **Node Specs:** 4 CPU, 8GB RAM per node
- **Storage:** Dynamic provisioning enabled
- **Networking:** CNI plugin installed (Calico/Flannel)
- **Ingress Controller:** nginx-ingress or traefik

### Additional Requirements
- **Prometheus Operator** (for monitoring)
- **Metrics Server** (for HPA)
- **Cert-Manager** (for TLS)
- **Container Registry** access (ghcr.io or private)

---

## 🏗️ Infrastructure Setup

### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

Verify:
```bash
kubectl get namespace tech-analysis-prod
```

### 2. Setup RBAC

```bash
kubectl apply -f k8s/rbac.yaml
```

Verify:
```bash
kubectl get serviceaccount -n tech-analysis-prod
kubectl get role,rolebinding -n tech-analysis-prod
```

### 3. Deploy Redis Cache

```bash
kubectl apply -f k8s/redis.yaml
```

Verify:
```bash
kubectl get pods -n tech-analysis-prod -l app=redis
kubectl exec -n tech-analysis-prod deploy/redis -- redis-cli ping
# Expected: PONG
```

### 4. Create Secrets

```bash
# Create from file
kubectl create secret generic technical-analysis-secret \
  --from-literal=SECRET_KEY=$(openssl rand -hex 32) \
  --from-literal=JWT_SECRET=$(openssl rand -hex 32) \
  -n tech-analysis-prod

# Or apply from secret.yaml (edit values first)
kubectl apply -f k8s/secret.yaml
```

Verify:
```bash
kubectl get secrets -n tech-analysis-prod
```

---

## 🚀 Deployment Steps

### Method 1: Direct Kubernetes Manifests

#### Step 1: Apply ConfigMaps
```bash
kubectl apply -f k8s/configmap.yaml
```

#### Step 2: Deploy Application
```bash
kubectl apply -f k8s/deployment.yaml
```

#### Step 3: Create Service
```bash
kubectl apply -f k8s/service.yaml
```

#### Step 4: Setup Ingress
```bash
# Edit ingress.yaml with your domain
kubectl apply -f k8s/ingress.yaml
```

#### Step 5: Enable Auto-Scaling
```bash
kubectl apply -f k8s/hpa.yaml
```

#### Step 6: Setup Monitoring
```bash
kubectl apply -f k8s/monitoring.yaml
```

### Method 2: Helm Chart (Recommended)

#### Step 1: Package Helm Chart
```bash
helm package helm/technical-analysis/
```

#### Step 2: Install with Helm
```bash
helm install technical-analysis ./technical-analysis-1.1.0.tgz \
  --namespace tech-analysis-prod \
  --create-namespace \
  --values helm/technical-analysis/values-production.yaml
```

#### Step 3: Verify Installation
```bash
helm list -n tech-analysis-prod
helm status technical-analysis -n tech-analysis-prod
```

---

## 📊 Monitoring & Observability

### 1. Prometheus Metrics

**Metrics Endpoint:** `http://<pod-ip>:8000/metrics`

**Key Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `ml_prediction_duration_seconds` - ML inference time
- `patterns_detected_total` - Pattern detection count
- `cache_hits_total` - Cache hits
- `cache_requests_total` - Cache requests

**Verify Metrics:**
```bash
# Port-forward to access metrics
kubectl port-forward -n tech-analysis-prod svc/technical-analysis 8000:8000

# Query metrics
curl http://localhost:8000/metrics | grep http_requests_total
```

### 2. Grafana Dashboards

**Access Grafana:**
```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open http://localhost:3000
```

**Import Dashboard:**
1. Go to Dashboards → Import
2. Load `k8s/monitoring.yaml` dashboard JSON
3. Select Prometheus data source

**Key Dashboard Panels:**
- Request rate (req/s)
- Response time P95/P99
- Error rate %
- ML inference latency
- Pattern detection count
- Cache hit rate
- CPU/Memory usage

### 3. Alerting

**Configured Alerts:**
- HighErrorRate (>5% for 5min)
- HighResponseTime (P95 >100ms for 10min)
- PodDown (>2min)
- HighCPUUsage (>80% for 10min)
- HighMemoryUsage (>85% for 10min)
- LowCacheHitRate (<50% for 15min)
- SlowMLInference (P95 >500ms for 10min)

**Test Alert:**
```bash
# Trigger high CPU alert
kubectl exec -n tech-analysis-prod deploy/technical-analysis -- \
  python -c "while True: pass"
```

### 4. Logs

**View Logs:**
```bash
# All pods
kubectl logs -n tech-analysis-prod -l app=technical-analysis --tail=100 -f

# Specific pod
kubectl logs -n tech-analysis-prod <pod-name> -f

# Previous pod (if crashed)
kubectl logs -n tech-analysis-prod <pod-name> --previous
```

**Structured Logging:**
All logs are in JSON format for easy parsing:
```json
{
  "event": "request_completed",
  "method": "POST",
  "path": "/api/v1/patterns/detect",
  "status_code": 200,
  "duration": "0.242s",
  "timestamp": "2025-11-12T10:30:45Z"
}
```

---

## ⚡ Scaling & Performance

### 1. Horizontal Pod Autoscaler (HPA)

**Current Configuration:**
- **Min Replicas:** 3
- **Max Replicas:** 50
- **CPU Target:** 70%
- **Memory Target:** 75%
- **Custom Metric:** 1000 req/s per pod

**Manual Scaling:**
```bash
# Scale to 10 replicas
kubectl scale deployment technical-analysis --replicas=10 -n tech-analysis-prod

# Check HPA status
kubectl get hpa -n tech-analysis-prod
kubectl describe hpa technical-analysis-hpa -n tech-analysis-prod
```

**Monitor Scaling:**
```bash
# Watch HPA in real-time
kubectl get hpa -n tech-analysis-prod -w

# Check pod count
kubectl get pods -n tech-analysis-prod -l app=technical-analysis
```

### 2. Vertical Pod Autoscaler (VPA) - Optional

**Install VPA:**
```bash
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vpa-v0.yaml
```

**Enable for Deployment:**
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: technical-analysis-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: technical-analysis
  updatePolicy:
    updateMode: "Auto"
```

### 3. Performance Benchmarks

**Expected Performance:**
- **Request Rate:** 1M+ req/s (with 50 pods)
- **Latency P95:** <1ms
- **Latency P99:** <10ms
- **ML Inference:** <250ms (with XGBoost v2)
- **Pattern Detection:** <250ms per request
- **Cache Hit Rate:** >85%

**Load Testing:**
```bash
# Install Locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://<your-domain>

# Or use k6
k6 run tests/load/k6-load-test.js
```

---

## 🔧 Troubleshooting

### 1. Pods Not Starting

**Check Pod Status:**
```bash
kubectl get pods -n tech-analysis-prod
kubectl describe pod <pod-name> -n tech-analysis-prod
```

**Common Issues:**
- **ImagePullBackOff:** Check image registry access
- **CrashLoopBackOff:** Check logs for startup errors
- **Pending:** Check resource availability

**Solutions:**
```bash
# Check events
kubectl get events -n tech-analysis-prod --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n tech-analysis-prod

# Check resource limits
kubectl describe nodes | grep -A 5 "Allocated resources"
```

### 2. Service Unreachable

**Check Service:**
```bash
kubectl get svc -n tech-analysis-prod
kubectl describe svc technical-analysis -n tech-analysis-prod
```

**Test Connectivity:**
```bash
# From within cluster
kubectl run -it --rm debug --image=alpine --restart=Never -n tech-analysis-prod -- sh
# Inside pod:
wget -qO- http://technical-analysis:8000/health

# Port-forward to local
kubectl port-forward svc/technical-analysis 8000:8000 -n tech-analysis-prod
curl http://localhost:8000/health
```

### 3. High Memory Usage

**Check Memory:**
```bash
kubectl top pods -n tech-analysis-prod
kubectl describe pod <pod-name> -n tech-analysis-prod | grep -A 5 "Limits"
```

**Solutions:**
- Increase memory limits in deployment.yaml
- Check for memory leaks in logs
- Review ML model memory usage
- Enable cache size limits

### 4. ML Model Loading Errors

**Check Logs:**
```bash
kubectl logs -n tech-analysis-prod -l app=technical-analysis | grep "ml_model"
```

**Common Issues:**
- Model file not found
- Insufficient memory
- Corrupted model file

**Solutions:**
```bash
# Check ConfigMap
kubectl get configmap ml-models-config -n tech-analysis-prod

# Recreate ConfigMap with model
kubectl create configmap ml-models-config \
  --from-file=ml_models/pattern_classifier_advanced_v2.pkl \
  -n tech-analysis-prod --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment
kubectl rollout restart deployment/technical-analysis -n tech-analysis-prod
```

### 5. Redis Connection Issues

**Check Redis:**
```bash
kubectl get pods -n tech-analysis-prod -l app=redis
kubectl exec -n tech-analysis-prod deploy/redis -- redis-cli ping
```

**Test from App:**
```bash
kubectl exec -it -n tech-analysis-prod deploy/technical-analysis -- \
  python -c "import redis; r=redis.Redis(host='redis-service', port=6379); print(r.ping())"
```

---

## 🔄 Disaster Recovery

### 1. Backup Strategy

**What to Backup:**
- Kubernetes manifests (Git)
- ConfigMaps and Secrets
- ML models
- Prometheus metrics (optional)
- Grafana dashboards

**Backup ConfigMaps:**
```bash
kubectl get configmap technical-analysis-config -n tech-analysis-prod -o yaml > backup/configmap.yaml
kubectl get secret technical-analysis-secret -n tech-analysis-prod -o yaml > backup/secret.yaml
```

### 2. Rollback Procedures

**Helm Rollback:**
```bash
# List releases
helm history technical-analysis -n tech-analysis-prod

# Rollback to previous version
helm rollback technical-analysis -n tech-analysis-prod

# Rollback to specific revision
helm rollback technical-analysis 2 -n tech-analysis-prod
```

**Kubectl Rollback:**
```bash
# Check rollout history
kubectl rollout history deployment/technical-analysis -n tech-analysis-prod

# Rollback to previous version
kubectl rollout undo deployment/technical-analysis -n tech-analysis-prod

# Rollback to specific revision
kubectl rollout undo deployment/technical-analysis --to-revision=2 -n tech-analysis-prod
```

### 3. Disaster Recovery Plan

**RTO:** 15 minutes  
**RPO:** 0 minutes (stateless service)

**Steps:**
1. Restore namespace: `kubectl apply -f k8s/namespace.yaml`
2. Restore secrets: `kubectl apply -f backup/secret.yaml`
3. Restore ConfigMaps: `kubectl apply -f backup/configmap.yaml`
4. Deploy Redis: `kubectl apply -f k8s/redis.yaml`
5. Deploy application: `kubectl apply -f k8s/deployment.yaml`
6. Restore service: `kubectl apply -f k8s/service.yaml`
7. Restore ingress: `kubectl apply -f k8s/ingress.yaml`
8. Verify health: `kubectl get pods -n tech-analysis-prod`

---

## 🔒 Security

### 1. Security Best Practices

✅ **Implemented:**
- Non-root containers (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Dropped all capabilities
- Network policies (optional)
- Pod Security Standards (restricted)
- RBAC with least privilege
- Secrets management
- TLS for ingress
- Rate limiting per IP

### 2. Security Scanning

**Scan Container Image:**
```bash
# Using Trivy
trivy image ghcr.io/gravitywavesml/gravity_techanalysis:v1.1.0

# Using Snyk
snyk container test ghcr.io/gravitywavesml/gravity_techanalysis:v1.1.0
```

**Scan Kubernetes Manifests:**
```bash
# Using kubesec
kubesec scan k8s/deployment.yaml

# Using kube-bench
kube-bench run --targets node,policies
```

### 3. Network Policies

**Apply Network Policy:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: technical-analysis-netpol
  namespace: tech-analysis-prod
spec:
  podSelector:
    matchLabels:
      app: technical-analysis
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 53  # DNS
        - protocol: UDP
          port: 53
```

---

## 📝 Maintenance

### 1. Rolling Updates

**Update Image:**
```bash
kubectl set image deployment/technical-analysis \
  technical-analysis=ghcr.io/gravitywavesml/gravity_techanalysis:v1.2.0 \
  -n tech-analysis-prod

# Watch rollout
kubectl rollout status deployment/technical-analysis -n tech-analysis-prod
```

**Update ConfigMap:**
```bash
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment/technical-analysis -n tech-analysis-prod
```

### 2. Health Checks

**Check All Components:**
```bash
# Pods
kubectl get pods -n tech-analysis-prod

# Services
kubectl get svc -n tech-analysis-prod

# Ingress
kubectl get ingress -n tech-analysis-prod

# HPA
kubectl get hpa -n tech-analysis-prod

# Health endpoint
curl http://<your-domain>/health
curl http://<your-domain>/health/ready
curl http://<your-domain>/health/live
```

### 3. Resource Cleanup

**Clean Old Resources:**
```bash
# Delete failed pods
kubectl delete pods --field-selector=status.phase=Failed -n tech-analysis-prod

# Clean completed jobs
kubectl delete jobs --field-selector=status.successful=1 -n tech-analysis-prod

# Remove old ReplicaSets
kubectl delete replicaset --all-namespaces --field-selector='status.replicas==0'
```

---

## 📞 Support

**Team:** DevOps & Cloud Infrastructure  
**Lead:** Lars Andersson  
**Slack:** #gravity-tech-devops  
**Email:** devops@gravity-tech.com  
**On-Call:** PagerDuty rotation

---

**Document Version:** 1.0  
**Last Updated:** November 12, 2025  
**Next Review:** December 12, 2025
