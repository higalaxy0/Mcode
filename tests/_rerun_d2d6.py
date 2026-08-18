"""Re-run D6 only to validate the peek_inbox race fix."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_teamagent_msg_matrix_real as M

M._reset_env()
M.test_d6_request_plan()

print("\nsummary:", sum(1 for _, c in M.results if c), "/", len(M.results))
for n, c in M.results:
    print(" ", "OK" if c else "XX", n)
