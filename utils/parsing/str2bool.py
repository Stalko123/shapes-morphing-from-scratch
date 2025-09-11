import argparse

def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    elif v.lower() in ("false", "0", "no", "n", "f"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}'")