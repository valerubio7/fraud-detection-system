import sys

# Remove MagicMock stubs so load tests use real packages.
# The root conftest stubs all heavy deps for unit tests; here we clear
# the ones actually needed by load test modules.
_STUB_ROOTS = {"psycopg2", "confluent_kafka", "fastavro"}
for _mod in list(sys.modules):
    root = _mod.split(".")[0]
    if root in _STUB_ROOTS:
        del sys.modules[_mod]
