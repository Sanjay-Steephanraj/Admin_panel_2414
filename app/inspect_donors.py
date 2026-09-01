import sys
import os

# Add app to path
sys.path.append(os.path.join(os.getcwd(), 'app'))

from tools.db_tool import execute_query

def inspect_donors():
    # Try common formats for '32'
    queries = [
        "SELECT donorid, firstname, lastname FROM sf_contacts WHERE donorid LIKE '%32%' LIMIT 5",
        "SELECT donorid, firstname, lastname FROM sf_contacts WHERE donorid = '000032'",
        "SELECT donorid, firstname, lastname FROM sf_contacts WHERE donorid = '#000032'",
        "SELECT donorid, firstname, lastname FROM sf_contacts LIMIT 5"
    ]
    
    for sql in queries:
        print(f"Executing: {sql}")
        rows, err = execute_query(sql)
        if err:
            print(f"Error: {err}")
        else:
            print(f"Rows: {rows}")
        print("-" * 30)

if __name__ == "__main__":
    inspect_donors()
