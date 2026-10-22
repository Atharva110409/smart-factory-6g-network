import os
import subprocess
import sys

def run_latex_build():
    miktex_bin = r"C:\Users\athar\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    if not os.path.exists(miktex_bin):
        print(f"Error: MiKTeX bin not found at {miktex_bin}")
        return False
        
    env = os.environ.copy()
    env["PATH"] = miktex_bin + os.pathsep + env.get("PATH", "")
    
    paper_dir = os.path.abspath("paper")
    
    pdflatex = os.path.join(miktex_bin, "pdflatex.exe")
    bibtex = os.path.join(miktex_bin, "bibtex.exe")
    
    commands = [
        ([pdflatex, "-interaction=nonstopmode", "main.tex"], "Pass 1: pdflatex"),
        ([bibtex, "main"], "Pass 2: bibtex"),
        ([pdflatex, "-interaction=nonstopmode", "main.tex"], "Pass 3: pdflatex"),
        ([pdflatex, "-interaction=nonstopmode", "main.tex"], "Pass 4: pdflatex final"),
    ]
    
    for cmd, desc in commands:
        print(f"\n{'='*20} {desc} {'='*20}")
        res = subprocess.run(cmd, cwd=paper_dir, env=env, capture_output=True, text=True)
        print(f"Return code: {res.returncode}")
        # Print relevant log snippets
        lines = res.stdout.splitlines()
        errors = [l for l in lines if l.startswith("!") or "error" in l.lower()]
        warnings = [l for l in lines if "warning" in l.lower() or "overfull" in l.lower()]
        
        if errors:
            print("Errors detected:")
            for e in errors[:20]:
                print("  ", e)
        if warnings:
            print(f"Warnings/Overfull count: {len(warnings)}")
            for w in warnings[:15]:
                print("  ", w)
                
        if res.returncode != 0 and "bibtex" not in desc:
            print("Failed! Showing last 30 lines of stdout:")
            print("\n".join(lines[-30:]))
            return False

    pdf_path = os.path.join(paper_dir, "main.pdf")
    if os.path.exists(pdf_path):
        print(f"\nSUCCESS: Generated {pdf_path} (size: {os.path.getsize(pdf_path):,} bytes)")
        return True
    else:
        print("\nFAILURE: main.pdf was not produced.")
        return False

if __name__ == "__main__":
    success = run_latex_build()
    sys.exit(0 if success else 1)
