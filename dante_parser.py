import configargparse

def get_parser():
    parser = configargparse.ArgumentParser(description="Incident Model Parser.")
    parser.add_argument("--train",
                        action="store_true",
                        default=True,
                        help="Enable training mode.")
    return parser