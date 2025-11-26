import csv
import subprocess
import json
import sys

CSV_PATH = "tests/test_matrix.csv"

def matches_expected(actual, expected):
    """
    Handles:
    - Single value (e.g., ENV_TARGETS)
    - Multi-value using "|" (e.g., ENV_POLICIES|ENV_TARGETS)
    - Wildcards ("ANY")
    - Global case for company
    """
    if expected in ["N/A", "", None]:
        return True

    if "|" in expected:
        allowed = [x.strip() for x in expected.split("|")]
        return actual in allowed

    if expected == "GLOBAL":
        return True

    return actual == expected


def run_test(row):
    test_id = row["Test ID"]
    cmd = row["CLI Command"]

    print(f"\n=== Running {test_id} ===")
    print(f"Command: {cmd}")

    # Run CLI command and capture output
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
    except Exception as e:
        print(f"Error running test {test_id}: {e}")
        return "FAIL"

    # Extract JSON from output
    try:
        json_start = output.index("{")
        json_end = output.rindex("}") + 1
        data = json.loads(output[json_start:json_end])
    except:
        print("Could not parse JSON output.")
        return "FAIL"

    # Expected values
    exp_intent = row["Expected Intent"].strip()
    exp_company = row["Expected Company"].strip()
    exp_section = row["Expected Section"].strip()
    exp_file_contains = row["Expected File Contains"].strip()

    fail = False

    # 1. Intent check — now supports multi-intent
    actual_intent = data.get("intent")
    if not matches_expected(actual_intent, exp_intent):
        print("Intent mismatch:", actual_intent)
        fail = True

    # 2. Company check — multi-company support
    actual_company = data.get("company")
    if not matches_expected(actual_company, exp_company):
        print("Company mismatch:", actual_company)
        fail = True

    # 3. Section check — multi-section support
    actual_sections = data.get("section_lookup", [])
    if exp_section not in ["ANY", "", "N/A"]:
        if "|" in exp_section:
            allowed_sections = [x.strip() for x in exp_section.split("|")]
            if not any(sec in allowed_sections for sec in actual_sections):
                print("Section mismatch:", actual_sections)
                fail = True
        else:
            if exp_section not in actual_sections:
                print("Section mismatch:", actual_sections)
                fail = True

    # 4. Snippet keyword check
    if exp_file_contains:
        keywords = [k.strip().lower() for k in exp_file_contains.split(";")]
        snippet_ok = False

        for res in data.get("results", []):
            text = res.get("text", "").lower()

            # A match happens if ANY keyword is found
            if any(k in text for k in keywords):
                snippet_ok = True
                break

        if not snippet_ok:
            print("Snippet keywords missing.")
            fail = True

    return "PASS" if not fail else "FAIL"


if __name__ == "__main__":
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = []

        for row in reader:
            result = run_test(row)
            results.append((row["Test ID"], result))

        print("\n=== TEST SUMMARY ===")
        for tid, r in results:
            print(f"{tid}: {r}")
