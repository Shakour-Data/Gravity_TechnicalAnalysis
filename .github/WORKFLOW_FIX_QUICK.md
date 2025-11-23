# 🚨 راهنمای سریع حل خطاهای CI/CD Workflow

**تاریخ:** 9 نوامبر 2025  
**وضعیت:** ⚠️ نیاز به تنظیمات GitHub  
**زمان حل:** 5 دقیقه

---

## 📋 خطاهای فعلی

```
1. ❌ Value 'dev' is not valid (line 166)
2. ❌ Value 'prod' is not valid (line 195)
3. ⚠️ Context access might be invalid: KUBE_CONFIG_DEV (line 178)
4. ⚠️ Context access might be invalid: KUBE_CONFIG_PROD (line 207)
```

## 🔍 علت

این خطاها به این دلیل هستند که **GitHub Environments** هنوز در repository ایجاد نشده‌اند. این مشکل فنی نیست، فقط نیاز به تنظیمات دارد.

---

## ✅ راه حل سریع (گزینه 1): غیرفعال کردن موقت

اگر الان نمی‌خواهید deployment را راه‌اندازی کنید، این تغییرات را اعمال کنید:

### کد فعلی را کامنت کنید:

```yaml
# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT JOBS (Temporarily Disabled)
# Uncomment and configure when ready for production deployment
# ═══════════════════════════════════════════════════════════════

# deploy-dev:
#   name: Deploy to Development
#   runs-on: ubuntu-latest
#   needs: build
#   if: github.ref == 'refs/heads/develop'
#   environment:
#     name: dev
#     url: https://dev.gravitytech.ai
#   steps:
#     - name: Checkout code
#       uses: actions/checkout@v4
#     
#     # Note: Create KUBE_CONFIG_DEV secret first
#     - name: Configure kubectl
#       uses: azure/k8s-set-context@v3
#       with:
#         method: kubeconfig
#         kubeconfig: ${{ secrets.KUBE_CONFIG_DEV }}
#     
#     - name: Deploy to Kubernetes
#       run: |
#         kubectl apply -f k8s/namespace.yaml
#         kubectl apply -f k8s/configmap.yaml
#         kubectl apply -f k8s/secret.yaml
#         kubectl apply -f k8s/deployment.yaml
#         kubectl apply -f k8s/service.yaml
#         kubectl rollout status deployment/technical-analysis -n tech-analysis-dev

# deploy-prod:
#   name: Deploy to Production
#   runs-on: ubuntu-latest
#   needs: build
#   if: github.event_name == 'release'
#   environment:
#     name: prod
#     url: https://api.gravitytech.ai
#   steps:
#     - name: Checkout code
#       uses: actions/checkout@v4
#     
#     # Note: Create KUBE_CONFIG_PROD secret first
#     - name: Configure kubectl
#       uses: azure/k8s-set-context@v3
#       with:
#         method: kubeconfig
#         kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}
#     
#     - name: Deploy to Kubernetes
#       run: |
#         kubectl apply -f k8s/namespace.yaml
#         kubectl apply -f k8s/configmap.yaml
#         kubectl apply -f k8s/secret.yaml
#         kubectl apply -f k8s/deployment.yaml
#         kubectl apply -f k8s/service.yaml
```

**این کار:**
- ✅ خطاها را برطرف می‌کند
- ✅ CI/CD همچنان کار می‌کند (test + build)
- ✅ فقط deployment غیرفعال می‌شود
- ✅ بعداً می‌توانید فعالش کنید

---

## 🎯 راه حل کامل (گزینه 2): تنظیم صحیح

اگر می‌خواهید deployment را فعال کنید:

### مرحله 1: ایجاد Environments در GitHub

1. به repository خود بروید: https://github.com/Shakour-Data/Gravity_TechAnalysis
2. **Settings** > **Environments** > **New environment**
3. دو environment ایجاد کنید:
   - نام: `dev`
   - نام: `prod`

### مرحله 2: اضافه کردن Secrets

1. **Settings** > **Secrets and variables** > **Actions**
2. **New repository secret**
3. دو secret اضافه کنید:

**KUBE_CONFIG_DEV:**
```bash
# در سرور development خود اجرا کنید:
kubectl config view --flatten --minify > kube-config-dev.yaml
# محتویات فایل را کپی کنید
```

**KUBE_CONFIG_PROD:**
```bash
# در سرور production خود اجرا کنید:
kubectl config view --flatten --minify > kube-config-prod.yaml
# محتویات فایل را کپی کنید
```

### مرحله 3: تست

پس از انجام مراحل بالا:
- خطاها در VSCode ناپدید می‌شوند ✅
- Workflow به درستی کار می‌کند ✅

---

## 📊 توصیه من

**برای الان (Day 2-3):**
```
✅ از گزینه 1 استفاده کنید (غیرفعال کردن deployment)
✅ روی development indicators تمرکز کنید
✅ بعداً deployment را راه‌اندازی کنید
```

**چرا؟**
1. شما در مرحله توسعه هستید (Day 2 از 7)
2. هنوز نیازی به production deployment ندارید
3. می‌توانید بعد از Day 7 deployment را تنظیم کنید

---

## 🚀 تصمیم شما؟

**گزینه A:** غیرفعال کردن موقت (5 دقیقه)
- من همین الان deployment jobs را کامنت می‌کنم
- خطاها برطرف می‌شوند
- فقط test + build کار می‌کند

**گزینه B:** تنظیم کامل (30 دقیقه)
- نیاز به دسترسی به GitHub Settings دارید
- نیاز به Kubernetes clusters دارید
- باید kubeconfig را تنظیم کنید

---

**کدام گزینه را ترجیح می‌دهید؟**

- اگر می‌خواهید سریع ادامه دهید → **گزینه A** ✅
- اگر آماده تنظیم کامل هستید → **گزینه B** 🔧

من منتظر تصمیم شما هستم! 🎯
