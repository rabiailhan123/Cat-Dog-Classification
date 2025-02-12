import logging

def setup_logging(log_file: str = "web.log") -> None:
    logging.basicConfig(
        level = logging.INFO,  # Log all INFO, WARNING, ERROR, CRITICAL
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename = log_file,
        filemode = "a",  # append to the log file in each run
    )
    # console handler to see logs in terminal too
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    """
    Log Levels

    DEBUG: Detailed information, typically useful for diagnosing problems.
    INFO: General events in the program.
    WARNING: Something unexpected, but the program continues to work.
    ERROR: A more serious problem, the program may not work properly.
    CRITICAL: A severe error, the program might stop.
    
    """