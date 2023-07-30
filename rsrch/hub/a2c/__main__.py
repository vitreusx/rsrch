from . import impl


def main():
    a2c = impl.A2C()
    a2c.train()


if __name__ == "__main__":
    main()
