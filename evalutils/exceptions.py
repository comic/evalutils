class EvalUtilsException(Exception):
    pass


class FileLoaderError(EvalUtilsException):
    pass


class ValidationError(EvalUtilsException):
    pass


class ConfigurationError(EvalUtilsException):
    pass
