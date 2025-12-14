import sys

sys.path.insert(0, 'src')
from gravity_tech.main import app

for r in sorted(app.routes, key=lambda r: r.path):
    print(sorted(r.methods), r.path)
