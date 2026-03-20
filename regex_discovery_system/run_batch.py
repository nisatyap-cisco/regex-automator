import requests
import time
import sys

BASE = "http://localhost:5001"

IDENTIFIERS = [
    "india aadhar card number",
    "india bank account number",
    "india bank routing number",
    "india corporate identification number",
    "india driver s license number",
    "india license plate number",
    "india national identity card number",
    "india phone number",
    "india post code",
    "india taxpayer id number",
    "india vat number",
    "ipv6",
    "java source code identification",
    "latvia postcode",
    "qatar passport number",
    "russian military identity number",
    "united arab emirates national identity card number",
    "usa medicaid number",
    "usa medicare number",
    "usa swift code",
]

for i, condition in enumerate(IDENTIFIERS, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/20] Starting: {condition}")
    print(f"{'='*60}")

    resp = requests.post(f"{BASE}/api/run", json={"condition": condition})
    if resp.status_code != 200:
        print(f"  ERROR starting: {resp.text}")
        continue

    run_id = resp.json()["run_id"]
    print(f"  Run ID: {run_id}")

    while True:
        time.sleep(5)
        try:
            status = requests.get(f"{BASE}/api/status/{run_id}").json()
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(5)
            continue

        step = status.get("step", "")
        st = status.get("status", "")

        if st == "done":
            print(f"  DONE — {status.get('overall', '')} (slug: {status.get('slug', '')})")
            break
        elif st == "error":
            print(f"  FAILED — {step}")
            # Get last log lines
            try:
                logs = requests.get(f"{BASE}/api/logs/{run_id}").json().get("logs", [])
                for line in logs[-3:]:
                    print(f"    {line}")
            except Exception:
                pass
            break
        else:
            print(f"  ... {step}")

print(f"\n{'='*60}")
print("ALL 20 RUNS COMPLETE")
print(f"{'='*60}")
