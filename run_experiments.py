import os
import subprocess

def run_experiment(script_name):
    """
    Runs a specific experiment script.

    Args:
        script_name (str): The Python script to execute.
    """
    print(f"Running: {script_name}")
    try:
        # Run the script
        subprocess.run(["python", script_name], check=True)
        print(f"Finished: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # List of scripts to run
    scripts = [
        "main.py",  # TRRL method
        "AIRL.py",  # Adversarial Inverse Reinforcement Learning
        "GAIL.py",  # Generative Adversarial Imitation Learning
        "BC.py",    # Behavior Cloning
        "Daggle.py",  # Daggle baseline
        "SQIL.py"   # Soft Q Imitation Learning
    ]

    # Run each script
    for script in scripts:
        run_experiment(script)

if __name__ == "__main__":
    main()
