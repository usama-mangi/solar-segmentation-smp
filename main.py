import utils.transport
import utils.session
import subprocess
import os
import pandas as pd

def run_experiments():
    DATA_DIR = "data/processed" 
    SSL_WEIGHTS_BASE = "weights"
    BASE_SAVE_DIR = "experiments"
    
    configs = [
        {"arch": "unet", "encoder": "simclr", "backbone": "resnet50"},
        {"arch": "fpn", "encoder": "simclr", "backbone": "resnet50"},
        {"arch": "pspnet", "encoder": "simclr", "backbone": "resnet50"},
        {"arch": "unet", "encoder": "moco", "backbone": "resnet50"},
        {"arch": "fpn", "encoder": "moco", "backbone": "resnet50"},
        {"arch": "pspnet", "encoder": "moco", "backbone": "resnet50"},
    ]
    
    results = []

    # Ensure python knows where src is
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src") + os.pathsep + env.get("PYTHONPATH", "")

    for conf in configs:
        arch = conf['arch']
        bb = conf['backbone']
        encoder = conf['encoder']
        run_name = f"{arch}_{encoder}_{bb}"
        save_path = os.path.join(BASE_SAVE_DIR, run_name)
        
        print(f"\n>>> Running Experiment: {run_name}")
        
        cmd = [
            "uv", "run", "python", "-m", "src.training.train_main",
            "--data_dir", DATA_DIR,
            "--save_dir", save_path,
            "--arch", arch,
            "--backbone", bb,
            "--weights", os.path.join(SSL_WEIGHTS_BASE, f"{encoder}_encoder_best.pth")
        ]
        
        try:
            # Pass the modified environment
            subprocess.run(cmd, check=True, env=env)
            
            metrics_path = os.path.join(save_path, "metrics.json")
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    best_dice = max(data['val_dice'])
                    results.append({"Architecture": arch, "Backbone": bb, "Best Val Dice": best_dice})
            else:
                results.append({"Architecture": arch, "Backbone": bb, "Best Val Dice": "Error"})
                
        except subprocess.CalledProcessError:
            print(f"Failed run: {run_name}")
            results.append({"Architecture": arch, "Backbone": bb, "Best Val Dice": "Failed"})

    print("\n\n=== Final Results ===")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    df.to_csv(os.path.join(BASE_SAVE_DIR, "comparison_results.csv"), index=False)

if __name__ == "__main__":
    run_experiments()