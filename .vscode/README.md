# VS Code Configuration for Gravity Technical Analysis

## Overview

این فولدر شامل تنظیمات VS Code برای پروژه Gravity Technical Analysis است.

## فایل‌ها

### `settings.json`
تنظیمات عمومی پروژه:
- Python testing با pytest
- Code formatting
- Type checking (strict mode)
- File exclusions (.pyc, __pycache__, etc.)

### `launch.json`
Debug configurations:
- **Python: Current File** - Debug فایل فعلی
- **Python: Debug Tests** - Debug یک فایل تست
- **Python: Debug TSE Tests** - Debug تست‌های TSE با coverage
- **Python: Debug All Tests** - Debug تمام تست‌ها

### `tasks.json`
Task definitions برای اجرای تست‌ها:
- **Run TSE Tests** - اجرای تست‌های TSE
- **Run TSE Tests with Coverage** - اجرای تست‌های TSE با coverage report
- **Run All Tests** - اجرای تمام تست‌ها
- **Run Unit Tests** - اجرای unit tests
- **Run API Tests** - اجرای API tests

### `extensions.json`
توصیه‌شده extension‌ها:
- Python
- Pylance
- Pytest
- Ruff
- GitLens
- GitHub Copilot

## استفاده

### اجرای تست‌های TSE در VS Code

#### روش 1: Testing Panel (پیشنهادی)
1. View → Testing یا `Ctrl+Shift+D` و انتخاب Testing
2. بر روی تست‌های TSE کلیک کنید
3. یا `Run` برای اجرا یا `Debug` برای debug

#### روش 2: Command Palette
1. `Ctrl+Shift+P` برای بازکردن Command Palette
2. "Run TSE Tests with Coverage" جستجو کنید
3. Enter

#### روش 3: Debug
1. `Ctrl+Shift+D` و انتخاب "Python: Debug TSE Tests"
2. `F5` برای شروع

### اجرای تمام تست‌ها

```
Ctrl+Shift+P → "Run All Tests" → Enter
```

یا

```
Ctrl+Shift+D → انتخاب "Python: Debug All Tests" → F5
```

## شناسایی خودکار تست‌ها

VS Code به طور خودکار تست‌های موجود در فولدر `tests/` را شناسایی می‌کند:

```
tests/
├── unit/          ✓ شناسایی شده
├── tse_data/      ✓ شناسایی شده
├── api/           ✓ شناسایی شده
├── services/      ✓ شناسایی شده
└── ...
```

## TSE Tests Testing Panel

در Testing panel، می‌توانید:
- **Run** - فقط اجرا کنید
- **Debug** - با breakpoint‌ها debug کنید
- **Run with Coverage** - coverage report دریافت کنید
- **Inspect** - جزئیات تست را ببینید

## Keyboard Shortcuts

| عملیات | Shortcut |
|--------|----------|
| Testing Panel | `Ctrl+Shift+D` |
| Command Palette | `Ctrl+Shift+P` |
| Debug | `F5` |
| Stop Debug | `Shift+F5` |
| Continue | `F5` |
| Step Over | `F10` |
| Step Into | `F11` |

## مشکلات معمول و حل‌ها

### مشکل: تست‌ها شناسایی نمی‌شوند
**حل:**
1. بررسی کنید که pytest نصب است: `pip install pytest`
2. پنجره VS Code را reload کنید: `Ctrl+Shift+P` → "Reload Window"
3. بررسی settings: `python.testing.pytestEnabled: true`

### مشکل: Coverage report خالی است
**حل:**
1. pytest-cov نصب کنید: `pip install pytest-cov`
2. دوباره تست را اجرا کنید

### مشکل: Imports کار نمی‌کند
**حل:**
1. فایل `__init__.py` در تمام فولدرها وجود دارد
2. Python interpreter را بررسی کنید: `Ctrl+Shift+P` → "Python: Select Interpreter"

## اطلاعات بیشتر

برای اطلاعات بیشتر:
- `tests/README.md` - تست‌ها
- `tests/TEST_STRUCTURE.md` - ساختار تست‌ها
- `TEST_EXECUTION_GUIDE.md` - راهنمای اجرا

---

**آخرین بروزرسانی**: 4 دسامبر 2025
