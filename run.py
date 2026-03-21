import sys
import subprocess
import platform
import os
import re

def get_nvidia_driver_version():
    """Versucht, die NVIDIA-Treiberversion via nvidia-smi zu ermitteln."""
    try:
        if platform.system() == "Windows":
            # Unter Windows ist nvidia-smi oft im Pfad, aber nicht immer.
            cmd = "nvidia-smi"
        else:
            cmd = "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
        
        output = subprocess.check_output(cmd, shell=True, text=True)
        
        if platform.system() == "Windows":
             # Parse output for Windows if complex, but usually query works same if flags supported
             # Fallback simple query
             output = subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True, text=True)
        
        version_str = output.strip().split('\n')[0]
        return version_str
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def check_cuda_availability():
    """Prüft, ob PyTorch CUDA verwenden kann."""
    try:
        import torch
        return torch.cuda.is_available(), torch.version.cuda
    except ImportError:
        return False, None

def install_torch_cuda(cuda_version="121"):
    """Installiert PyTorch mit CUDA-Support."""
    print(f"🔄 Installiere PyTorch mit CUDA {cuda_version} Support...")
    
    # Deinstalliere existierendes Torch um Konflikte zu vermeiden
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
    
    # Installationsbefehl zusammenbauen
    # Standard: Stable (2.x)
    index_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", index_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Installation abgeschlossen.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei der Installation: {e}")
        return False

def install_requirements_excluding_torch():
    """Installiert alle Requirements außer Torch-Pakete."""
    print("📦 Überprüfe und installiere Abhängigkeiten...")
    
    try:
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        
        # Filter torch packages
        pkgs = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        non_torch_pkgs = [p for p in pkgs if not any(x in p.lower() for x in ['torch', 'torchvision', 'torchaudio'])]
        
        if non_torch_pkgs:
            # Erstelle temporäre requirements datei oder installiere direkt
            # Direkt installieren ist einfacher
            cmd = [sys.executable, "-m", "pip", "install"] + non_torch_pkgs
            subprocess.run(cmd, check=True)
            return True
    except Exception as e:
        print(f"⚠️ Warnung bei der Installation der Abhängigkeiten: {e}")
        # Fallback: Versuche normales install
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=False)
        return False
    return True

def get_gpu_info():
    """Versucht, die NVIDIA-Treiberversion und Compute Capability zu ermitteln."""
    try:
        cmd = "nvidia-smi --query-gpu=driver_version,compute_cap,name --format=csv,noheader"
        output = subprocess.check_output(cmd, shell=True, text=True)
        lines = output.strip().split('\n')
        if not lines:
            return None, None, None
            
        # Nimm die erste GPU
        parts = [p.strip() for p in lines[0].split(',')]
        driver = parts[0]
        compute_cap = parts[1]
        name = parts[2]
        return driver, compute_cap, name
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None, None, None

def setup_environment():
    print("🚀 Starte automatische System-Einrichtung...")
    
    # 1. Hardware Check
    driver_version, compute_cap, gpu_name = get_gpu_info()
    
    # 2. Installiere allgemeine Abhängigkeiten (ohne Torch)
    install_requirements_excluding_torch()

    if not driver_version:
        print("⚠️ Keine NVIDIA-Grafikkarte oder Treiber gefunden (nvidia-smi fehlgeschlagen).")
        print("   Das System wird im CPU-Modus laufen.")
        # Stelle sicher dass Torch (CPU) installiert ist
        subprocess.run([sys.executable, "-m", "pip", "install", "torch>=2.1.0", "torchvision", "torchaudio"], check=False)
        return

    print(f"✅ NVIDIA GPU gefunden: {gpu_name}")
    print(f"✅ Treiber: {driver_version}, Compute Capability: {compute_cap}")
    
    # Version parsing
    try:
        major_ver = int(driver_version.split('.')[0])
        cc_major = int(compute_cap.split('.')[0])
    except:
        major_ver = 0
        cc_major = 0
        
    target_cuda = None
    
    # Architektur-Check:
    # Ältere Karten (Kepler, Maxwell, Pascal teils) profitieren oft von CUDA 11.8
    if cc_major < 6: # Kepler (3.x), Maxwell (5.x)
        target_cuda = "118"
        print("💡 Ältere GPU Architektur erkannt. Verwende CUDA 11.8 für bessere Kompatibilität.")
    elif platform.system() == "Linux":
        if major_ver >= 525:
            target_cuda = "121" # CUDA 12.1
        elif major_ver >= 450:
            target_cuda = "118" # CUDA 11.8
    elif platform.system() == "Windows":
        if major_ver >= 500:
            target_cuda = "121"
        else:
            target_cuda = "118"
            
    if not target_cuda:
        print("⚠️ Treiber-Version konnte nicht automatisch zugeordnet werden.")
        # Fallback auf Standard-Torch
        subprocess.run([sys.executable, "-m", "pip", "install", "torch>=2.1.0", "torchvision", "torchaudio"], check=False)
        return

    # 3. Check current Torch status
    cuda_available, current_cuda_version = check_cuda_availability()
    
    if cuda_available:
        print(f"✅ PyTorch CUDA ist bereits aktiv (Version {current_cuda_version}).")
        return

    print(f"⚠️ GPU gefunden, aber PyTorch CUDA ist nicht aktiv.")
    print(f"   Ziel-CUDA Version: {target_cuda}")
    
    # 4. Installiere korrekte Version
    install_torch_cuda(target_cuda)

def check_venv():
    """Prüft, ob das Skript in einem Virtual Environment läuft."""
    # Check for venv/virtualenv
    in_venv = (sys.prefix != sys.base_prefix) or hasattr(sys, 'real_prefix')
    
    if not in_venv:
        print("⚠️  HINWEIS: Du scheinst nicht in einem Virtual Environment (venv) zu sein.")
        print("   Es wird empfohlen, ein venv zu erstellen, um Konflikte mit System-Paketen zu vermeiden:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  (Linux/Mac)")
        print("   venv\\Scripts\\activate    (Windows)")
        print("-" * 50)

def main():
    check_venv()
    
    # Setup durchführen
    setup_environment()
    
    print("\n" + "="*50)
    print("🏁 Setup beendet. Starte Anwendung...")
    print("="*50 + "\n")
    
    # App starten
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\nBeendet.")

if __name__ == "__main__":
    main()
