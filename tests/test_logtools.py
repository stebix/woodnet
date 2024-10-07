from woodnet.logtools.infrastructure import create_instance_logger


class MockObjectForInstanceLogger:

    def __init__(self) -> None:
        pass


def test_create_instance_logger():
    obj = MockObjectForInstanceLogger()
    logger = create_instance_logger(obj)
    print(logger.name)