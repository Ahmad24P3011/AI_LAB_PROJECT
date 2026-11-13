"""Check that every required package can be imported, and print its version."""
import importlib

# (name used in pip, name used in `import`)
PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("streamlit", "streamlit"),
    ("requests", "requests"),
    ("jupyter", "jupyter_core"),  # jupyter itself is a bundle; jupyter_core is its base
]


def main():
    missing = []
    for pip_name, import_name in PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown version")
            print(f"{pip_name:14} OK   {version}")
        except ImportError:
            print(f"{pip_name:14} MISSING")
            missing.append(pip_name)

    if missing:
        print("\nSetup NOT complete. Missing:", ", ".join(missing))
    else:
        print("\nSetup OK")


if __name__ == "__main__":
    main()
