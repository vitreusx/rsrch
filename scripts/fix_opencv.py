import subprocess
import re
import shlex


def main():
    out = subprocess.check_output(["pip", "freeze"], encoding="utf-8")

    m = re.search(r"opencv-python([^=]*)==(?P<version>[0-9\.]*)", out)
    if m is not None:
        cmd = ["pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"]
        subprocess.run(cmd)

        version = m.group("version")
        cmd = ["pip", "install", "-U", f"opencv-python-headless==${version}"]
        subprocess.run(cmd)
