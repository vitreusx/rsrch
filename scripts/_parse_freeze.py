import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("packages", nargs="*")
    
    args = p.parse_args()
    for pkg in args.packages:
        name, ver = pkg.split("==")
        print(name)
        print(pkg)

if __name__ == "__main__":
    main()