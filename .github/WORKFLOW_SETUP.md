# GitHub Actions Workflow Setup Guide

This guide explains how to configure the CI/CD pipeline for the Gravity Technical Analysis project.

## ⚠️ Current Status

The CI/CD workflow (`.github/workflows/ci-cd.yml`) is configured but requires setup in your GitHub repository settings. Until setup is complete, you'll see validation warnings - **this is normal and expected**.

## 📋 Setup Steps

### 1. Create GitHub Environments

GitHub Environments allow you to configure deployment protection rules and secrets per environment.

**Steps:**
1. Go to your repository on GitHub
2. Click **Settings** (top menu)
3. Click **Environments** (left sidebar)
4. Click **New environment**

**Create these environments:**

#### Environment: `dev`
- **Name:** `dev`
- **URL:** `https://dev.gravitytech.ai` (optional)
- **Protection rules:** None (auto-deploy)
- **Secrets:** Will add in step 2

#### Environment: `prod`
- **Name:** `prod`
- **URL:** `https://api.gravitytech.ai` (optional)
- **Protection rules:** 
  - ✅ Required reviewers (add team members)
  - ✅ Wait timer: 5 minutes (optional)
- **Secrets:** Will add in step 2

### 2. Add Repository Secrets

Secrets are encrypted environment variables used in workflows.

**Steps:**
1. Go to **Settings > Secrets and variables > Actions**
2. Click **New repository secret**
3. Add each secret below:

#### Required Secrets:

**`KUBE_CONFIG_DEV`**
- **Purpose:** Kubernetes cluster config for development
- **How to get:**
  ```bash
  # From your dev cluster
  kubectl config view --flatten --minify > kube-config-dev.yaml
  # Copy the contents of kube-config-dev.yaml
  ```
- **Value:** Paste the entire kubeconfig file content

**`KUBE_CONFIG_PROD`**
- **Purpose:** Kubernetes cluster config for production
- **How to get:**
  ```bash
  # From your prod cluster
  kubectl config view --flatten --minify > kube-config-prod.yaml
  # Copy the contents of kube-config-prod.yaml
  ```
- **Value:** Paste the entire kubeconfig file content

**`DOCKER_USERNAME`** (if using Docker Hub)
- **Purpose:** Docker Hub login for pushing images
- **Value:** Your Docker Hub username

**`DOCKER_PASSWORD`** (if using Docker Hub)
- **Purpose:** Docker Hub authentication
- **Value:** Your Docker Hub access token (not password)
- **How to get:**
  1. Go to Docker Hub > Account Settings > Security
  2. Click "New Access Token"
  3. Copy the token

### 3. Verify Setup

After completing steps 1-2:

1. The validation errors in VSCode should disappear
2. Push a commit to the `develop` branch to test dev deployment
3. Create a release to test prod deployment

## 🔄 Workflow Behavior

### On Push to `main` or `develop`:
- ✅ Runs all tests
- ✅ Runs linting
- ✅ Builds Docker image
- ✅ Deploys to **dev** environment (if on `develop` branch)

### On Pull Request:
- ✅ Runs all tests
- ✅ Runs linting
- ❌ Does NOT deploy

### On Release Published:
- ✅ Runs all tests
- ✅ Runs linting
- ✅ Builds Docker image
- ✅ Deploys to **prod** environment (with approval if configured)

## 🛠️ Alternative: Disable Deployment Jobs

If you don't want to set up environments yet, you can temporarily disable deployment:

**Option 1: Comment out deployment jobs**
```yaml
# Uncomment when ready to deploy
# deploy-dev:
#   name: Deploy to Development
#   ...
```

**Option 2: Skip on missing secrets**
Add this condition to deployment jobs:
```yaml
if: github.ref == 'refs/heads/develop' && secrets.KUBE_CONFIG_DEV != ''
```

## 📚 Additional Resources

- [GitHub Environments Documentation](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Kubernetes kubectl Config](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/)

## ❓ Troubleshooting

### Error: "Environment not found"
- **Cause:** Environment not created in repository settings
- **Fix:** Follow Step 1 above

### Error: "Secret KUBE_CONFIG_DEV not found"
- **Cause:** Secret not added to repository
- **Fix:** Follow Step 2 above

### Error: "Invalid kubeconfig"
- **Cause:** Malformed kubeconfig file
- **Fix:** Ensure you copied the entire file content, including:
  ```yaml
  apiVersion: v1
  kind: Config
  clusters:
    ...
  ```

### VSCode Still Shows Errors
- **Cause:** VSCode validation is checking if environments exist
- **Status:** This is a warning, not an error
- **Impact:** None - workflow will work once environments are created
- **Fix:** Create environments in GitHub (Step 1)

## ✅ Verification Checklist

Before pushing to production:

- [ ] `dev` environment created
- [ ] `prod` environment created with reviewers
- [ ] `KUBE_CONFIG_DEV` secret added
- [ ] `KUBE_CONFIG_PROD` secret added
- [ ] Test deployment to dev successful
- [ ] Test deployment to prod successful (via release)
- [ ] Rollback procedure tested
- [ ] Monitoring configured
- [ ] Alerts configured

---

**Document Version:** 1.0.0  
**Last Updated:** November 8, 2025  
**Related:** `.github/workflows/ci-cd.yml`
