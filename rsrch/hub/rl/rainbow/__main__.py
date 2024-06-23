from .runner import Runner

if __name__ == "__main__":
    # This indirection is required due to the fact that subprocesses cannot
    # access functions defined in the main script.
    Runner.main()
