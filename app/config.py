# config.py — Central configuration for the LLM Guardrails Firewall
# Adjust these values to tune the system's sensitivity

# Decision threshold: confidence above this value triggers a block
THRESHOLD = 0.70

# Categories the classifier recognises
CATEGORIES = ["jailbreak", "injection", "toxic", "harmful", "safe"]

# Label constants
LABEL_SAFE   = "Safe"
LABEL_UNSAFE = "Unsafe"

# Paths
TEST_SUITE_PATH  = "data/test_suite.json"
LOGS_PATH        = "results/logs.csv"
METRICS_PATH     = "results/metrics.json"
GRAPHS_DIR       = "results/graphs"

# Logging verbosity (set False to silence per-prompt prints)
VERBOSE = True
