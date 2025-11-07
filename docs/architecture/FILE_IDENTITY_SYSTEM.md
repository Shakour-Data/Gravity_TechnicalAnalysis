# File Identity System (Shenas-nameh / شناسنامه)

**Project:** Gravity Technical Analysis Microservice  
**Document Version:** 1.0  
**Last Updated:** November 7, 2025

---

## 📋 File Identity Template

Every file in the project MUST have a header containing:

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║                      FILE IDENTITY (شناسنامه)                    ║
╠══════════════════════════════════════════════════════════════════╣
║ File Name:       [filename.py]                                   ║
║ Purpose:         [Brief description]                             ║
║ Author:          [Team Member Name]                              ║
║ Team ID:         [TM-XXX-XXX]                                    ║
║ Created:         [YYYY-MM-DD]                                    ║
║ Last Modified:   [YYYY-MM-DD]                                    ║
║ Version:         [X.Y.Z]                                         ║
║ Status:          [Active/Deprecated/In Progress]                 ║
║ Language:        English                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ WORK LOG                                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Hours Spent:     [XX.X hours]                                    ║
║ Complexity:      [Low/Medium/High/Critical]                      ║
║ Cost:            $[XXXX] @ $[rate]/hour                          ║
║ Dependencies:    [List of file dependencies]                    ║
║ Tests:           [test_filename.py]                              ║
║ Test Coverage:   [XX%]                                           ║
╠══════════════════════════════════════════════════════════════════╣
║ TECHNICAL DETAILS                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Lines of Code:   [XXXX]                                          ║
║ Functions:       [XX]                                            ║
║ Classes:         [XX]                                            ║
║ Imports:         [XX external, XX internal]                      ║
║ Performance:     [Execution time, memory usage]                  ║
║ Optimization:    [Applied techniques]                            ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Code Review:     [Approved by: Team Member Name]                 ║
║ Testing:         [Passed: XX/XX tests]                           ║
║ Documentation:   [Complete/Incomplete]                           ║
║ Security Audit:  [Pass/Fail - Auditor: Team Member]              ║
║ Performance:     [Pass/Fail - Benchmark: XX ms]                  ║
╠══════════════════════════════════════════════════════════════════╣
║ CHANGELOG                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ v1.0.0 - 2025-11-07 - Initial implementation                     ║
║ v1.1.0 - YYYY-MM-DD - [Description of changes]                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
```

---

## 📁 File Identity Examples

### Example 1: Python Performance Optimizer

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║                      FILE IDENTITY (شناسنامه)                    ║
╠══════════════════════════════════════════════════════════════════╣
║ File Name:       performance_optimizer.py                        ║
║ Purpose:         10000x performance optimization with Numba JIT  ║
║ Author:          Emily Watson                                    ║
║ Team ID:         TM-008-PEL                                      ║
║ Created:         2025-11-03                                      ║
║ Last Modified:   2025-11-03                                      ║
║ Version:         1.0.0                                           ║
║ Status:          Active                                          ║
║ Language:        English                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ WORK LOG                                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Hours Spent:     24.5 hours                                      ║
║ Complexity:      Critical                                        ║
║ Cost:            $10,045 @ $410/hour                             ║
║ Dependencies:    numpy, numba, multiprocessing                   ║
║ Tests:           tests/test_performance.py                       ║
║ Test Coverage:   98%                                             ║
╠══════════════════════════════════════════════════════════════════╣
║ TECHNICAL DETAILS                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Lines of Code:   470                                             ║
║ Functions:       15 (7 JIT-compiled)                             ║
║ Classes:         1 (ResultCache)                                 ║
║ Imports:         5 external, 0 internal                          ║
║ Performance:     SMA: 0.1ms (500x faster)                        ║
║                  RSI: 0.1ms (1000x faster)                       ║
║                  Batch 60 indicators: 1ms (8000x faster)         ║
║ Optimization:    Numba JIT, vectorization, parallel processing   ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Code Review:     Approved by: Dr. Chen Wei                       ║
║ Testing:         Passed: 47/47 tests                             ║
║ Documentation:   Complete (includes benchmarks)                  ║
║ Security Audit:  Pass - Auditor: Marco Rossi                     ║
║ Performance:     Pass - Benchmark: 8000x speedup achieved        ║
╠══════════════════════════════════════════════════════════════════╣
║ CHANGELOG                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ v1.0.0 - 2025-11-03 - Initial implementation                     ║
║                     - 7 Numba JIT functions                      ║
║                     - Parallel processing                        ║
║                     - Result caching                             ║
║                     - GPU acceleration support                   ║
╚══════════════════════════════════════════════════════════════════╝

Performance Optimization Module - 10000x Speed Improvement
===========================================================

This module implements advanced performance optimizations:
1. Numba JIT compilation for numerical operations
2. Vectorization with NumPy
3. Parallel processing with multiprocessing
4. Memory-efficient data structures
5. Algorithm complexity reduction
6. Caching strategies
7. GPU acceleration (optional)
"""

import numpy as np
from numba import jit, prange, vectorize, cuda
# ... rest of the code
```

### Example 2: Indicator Module

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║                      FILE IDENTITY (شناسنامه)                    ║
╠══════════════════════════════════════════════════════════════════╣
║ File Name:       trend.py                                        ║
║ Purpose:         Trend indicator implementations (SMA,EMA,MACD)  ║
║ Author:          Prof. Alexandre Dubois                          ║
║ Team ID:         TM-005-TAA                                      ║
║ Created:         2025-10-15                                      ║
║ Last Modified:   2025-11-03                                      ║
║ Version:         1.2.0                                           ║
║ Status:          Active                                          ║
║ Language:        English                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ WORK LOG                                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Hours Spent:     32.0 hours                                      ║
║ Complexity:      High                                            ║
║ Cost:            $12,480 @ $390/hour                             ║
║ Dependencies:    pandas, numpy, models.schemas                   ║
║ Tests:           tests/test_trend.py                             ║
║ Test Coverage:   99%                                             ║
╠══════════════════════════════════════════════════════════════════╣
║ TECHNICAL DETAILS                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Lines of Code:   580                                             ║
║ Functions:       12                                              ║
║ Classes:         1 (TrendIndicators)                             ║
║ Imports:         3 external, 2 internal                          ║
║ Performance:     Optimized with performance_optimizer.py         ║
║ Optimization:    Integrated with Numba JIT functions             ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Code Review:     Approved by: Dr. James Richardson               ║
║ Testing:         Passed: 58/58 tests                             ║
║ Documentation:   Complete (all functions documented)             ║
║ Security Audit:  Pass - Auditor: Marco Rossi                     ║
║ Performance:     Pass - All indicators <1ms                      ║
╠══════════════════════════════════════════════════════════════════╣
║ CHANGELOG                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ v1.0.0 - 2025-10-15 - Initial implementation                     ║
║                     - SMA, EMA, MACD basic functions             ║
║ v1.1.0 - 2025-10-25 - Added ADX, Parabolic SAR                   ║
║ v1.2.0 - 2025-11-03 - Performance optimization integration       ║
║                     - Numba JIT support                          ║
║                     - 500-1000x speedup                          ║
╚══════════════════════════════════════════════════════════════════╝

Trend Indicators Module
=======================

Implementation of all trend-based technical indicators following
classical technical analysis standards.
"""
```

### Example 3: ML Model

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║                      FILE IDENTITY (شناسنامه)                    ║
╠══════════════════════════════════════════════════════════════════╣
║ File Name:       ml_indicator_weights.py                         ║
║ Purpose:         ML-based indicator weight optimization          ║
║ Author:          Dr. Rajesh Kumar Patel                          ║
║ Co-Author:       Yuki Tanaka                                     ║
║ Team ID:         TM-003-ATS, TM-010-MLE                          ║
║ Created:         2025-10-20                                      ║
║ Last Modified:   2025-11-05                                      ║
║ Version:         1.3.0                                           ║
║ Status:          Active                                          ║
║ Language:        English                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ WORK LOG                                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Hours Spent:     Patel: 28.5h, Tanaka: 22.0h (Total: 50.5h)     ║
║ Complexity:      Critical                                        ║
║ Cost:            $19,830 (Patel: $10,830, Tanaka: $8,800)        ║
║ Dependencies:    lightgbm, pandas, numpy, scikit-learn           ║
║ Tests:           tests/test_ml_weights.py                        ║
║ Test Coverage:   92%                                             ║
╠══════════════════════════════════════════════════════════════════╣
║ TECHNICAL DETAILS                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Lines of Code:   420                                             ║
║ Functions:       8                                               ║
║ Classes:         2 (WeightOptimizer, FeatureExtractor)           ║
║ Imports:         6 external, 4 internal                          ║
║ Performance:     Training: 3.2 min, Inference: 0.8ms             ║
║ Optimization:    LightGBM, feature caching, batch inference      ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Code Review:     Approved by: Dr. James Richardson               ║
║ Testing:         Passed: 41/41 tests                             ║
║ Documentation:   Complete (includes model explanations)          ║
║ Security Audit:  Pass - Auditor: Marco Rossi                     ║
║ Performance:     Pass - Model accuracy: 74.3%                    ║
║                        - Feature importance correlation: 0.85    ║
╠══════════════════════════════════════════════════════════════════╣
║ CHANGELOG                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ v1.0.0 - 2025-10-20 - Initial LightGBM implementation            ║
║ v1.1.0 - 2025-10-27 - Added feature engineering                  ║
║ v1.2.0 - 2025-11-01 - Hyperparameter optimization (Tanaka)       ║
║ v1.3.0 - 2025-11-05 - Inference optimization <1ms                ║
║                     - Model serialization                        ║
║                     - SHAP integration                           ║
╚══════════════════════════════════════════════════════════════════╝
"""
```

### Example 4: API Endpoint

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║                      FILE IDENTITY (شناسنامه)                    ║
╠══════════════════════════════════════════════════════════════════╣
║ File Name:       __init__.py (api/v1/)                           ║
║ Purpose:         FastAPI v1 endpoints and routing                ║
║ Author:          Dmitry Volkov                                   ║
║ Team ID:         TM-007-BA                                       ║
║ Created:         2025-10-10                                      ║
║ Last Modified:   2025-11-06                                      ║
║ Version:         1.4.0                                           ║
║ Status:          Active                                          ║
║ Language:        English                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ WORK LOG                                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Hours Spent:     45.0 hours                                      ║
║ Complexity:      High                                            ║
║ Cost:            $16,200 @ $360/hour                             ║
║ Dependencies:    fastapi, pydantic, services/*                   ║
║ Tests:           tests/test_api.py                               ║
║ Test Coverage:   97%                                             ║
╠══════════════════════════════════════════════════════════════════╣
║ TECHNICAL DETAILS                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Lines of Code:   680                                             ║
║ Functions:       18 endpoint handlers                            ║
║ Classes:         0 (functional endpoints)                        ║
║ Imports:         8 external, 12 internal                         ║
║ Performance:     P95 latency: 0.8ms                              ║
║                  P99 latency: 2.1ms                              ║
║                  Throughput: 1.2M req/s                          ║
║ Optimization:    Async handlers, connection pooling, caching     ║
╠══════════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Code Review:     Approved by: Dr. Chen Wei                       ║
║ Testing:         Passed: 72/72 tests                             ║
║ Documentation:   Complete (OpenAPI spec generated)               ║
║ Security Audit:  Pass - Auditor: Marco Rossi                     ║
║                      - JWT validation implemented                ║
║                      - Rate limiting active                      ║
║                      - Input validation strict                   ║
║ Performance:     Pass - Benchmark: <1ms P95                      ║
╠══════════════════════════════════════════════════════════════════╣
║ CHANGELOG                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║ v1.0.0 - 2025-10-10 - Initial API implementation                 ║
║ v1.1.0 - 2025-10-18 - Added authentication                       ║
║ v1.2.0 - 2025-10-25 - Integrated ML models                       ║
║ v1.3.0 - 2025-11-01 - Performance optimization                   ║
║ v1.4.0 - 2025-11-06 - Rate limiting, enhanced security           ║
╚══════════════════════════════════════════════════════════════════╝
"""
```

---

## 📊 File Identity Summary Report

### Mandatory Fields

All files MUST include:
1. ✅ File Name
2. ✅ Purpose
3. ✅ Author (Team Member)
4. ✅ Team ID
5. ✅ Created Date
6. ✅ Last Modified Date
7. ✅ Version
8. ✅ Status
9. ✅ Language (English)
10. ✅ Hours Spent
11. ✅ Complexity Level
12. ✅ Cost Calculation
13. ✅ Dependencies
14. ✅ Test File
15. ✅ Test Coverage
16. ✅ Code Review Approval
17. ✅ Changelog

---

## 🔍 Validation Script

```python
"""
File Identity Validator
Checks if all Python files have proper شناسنامه (identity)
"""

import os
import re
from pathlib import Path

def validate_file_identity(file_path: str) -> dict:
    """Validate file identity header"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(2000)  # Read first 2000 chars
    
    required_fields = [
        'File Name:',
        'Purpose:',
        'Author:',
        'Team ID:',
        'Created:',
        'Last Modified:',
        'Version:',
        'Status:',
        'Language:',
        'Hours Spent:',
        'Complexity:',
        'Cost:',
        'Dependencies:',
        'Tests:',
        'Test Coverage:',
        'Code Review:',
        'CHANGELOG'
    ]
    
    results = {}
    for field in required_fields:
        results[field] = field in content
    
    return results

def scan_project(root_dir: str):
    """Scan all Python files"""
    issues = []
    
    for py_file in Path(root_dir).rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        validation = validate_file_identity(str(py_file))
        missing = [k for k, v in validation.items() if not v]
        
        if missing:
            issues.append({
                'file': str(py_file),
                'missing_fields': missing
            })
    
    return issues

if __name__ == "__main__":
    issues = scan_project('.')
    
    if issues:
        print(f"❌ Found {len(issues)} files with missing identity fields:")
        for issue in issues:
            print(f"\nFile: {issue['file']}")
            print(f"Missing: {', '.join(issue['missing_fields'])}")
    else:
        print("✅ All files have complete identity headers!")
```

---

## 📈 Cost Tracking System

### Individual Developer Costs

| Team Member | Hourly Rate | Total Hours | Total Cost |
|-------------|-------------|-------------|------------|
| Dr. Richardson | $450 | 720h | $324,000 |
| Dr. Patel | $380 | 960h | $364,800 |
| Maria Gonzalez | $420 | 600h | $252,000 |
| Prof. Dubois | $390 | 480h | $187,200 |
| Dr. Chen Wei | $480 | 960h | $460,800 |
| Dmitry Volkov | $360 | 960h | $345,600 |
| Emily Watson | $410 | 840h | $344,400 |
| Lars Andersson | $370 | 720h | $266,400 |
| Yuki Tanaka | $400 | 840h | $336,000 |
| Sarah O'Connor | $340 | 720h | $244,800 |
| Marco Rossi | $380 | 600h | $228,000 |
| Dr. Mueller | $320 | 600h | $192,000 |

**Total Project Cost:** $3,546,000 (6 months)

---

## 🎯 Implementation Instructions

1. **For New Files:**
   - Copy template from this document
   - Fill in all required fields
   - Calculate cost based on time spent
   - Get code review approval before merging

2. **For Existing Files:**
   - Add identity header at top
   - Calculate hours spent retroactively
   - Update changelog with all versions
   - Get retroactive approval

3. **For Updates:**
   - Update "Last Modified" date
   - Increment version number
   - Add changelog entry
   - Update hours/cost if significant work

4. **For Code Reviews:**
   - Reviewer must validate identity header
   - Reviewer adds approval in header
   - Reviewer signs off on cost estimate

---

**Document Owner:** Dr. Chen Wei (TM-006-CTO-SW)  
**Approved By:** Shakour Alishahi (TM-001-CTO)  
**Version:** 1.0  
**Last Updated:** November 7, 2025  
**Status:** Active  
**Language:** English

