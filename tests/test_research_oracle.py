import unittest

from research_impl.oracle import run_oracle_suite


class OracleTests(unittest.TestCase):
    def test_all_deletion_cases(self):
        self.assertTrue(run_oracle_suite()["passed"])


if __name__ == "__main__":
    unittest.main()
