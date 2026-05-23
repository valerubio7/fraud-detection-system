import sys

for _mod in list(sys.modules):
    if _mod.startswith("psycopg2"):
        del sys.modules[_mod]
