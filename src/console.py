class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_info(text: str):
    print(f"{Colors.CYAN}{text}{Colors.RESET}")


def print_debug(text: str):
    print(f"{Colors.MAGENTA}{text}{Colors.RESET}")


def print_success(text: str):
    print(f"{Colors.GREEN}{text}{Colors.RESET}")


def print_error(text: str):
    print(f"{Colors.RED}{text}{Colors.RESET}")


def print_warn(text: str):
    print(f"{Colors.YELLOW}{text}{Colors.RESET}")
