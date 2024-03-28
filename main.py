# runner.py
import subprocess

def run_scripts():
    # Execute script1.py
    subprocess.run(["python", "Data_extraction.py"])

    # Execute script2.py
    subprocess.run(["python", "model.py"])

if __name__ == "__main__":
    run_scripts()
