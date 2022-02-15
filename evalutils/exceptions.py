class EvalUtilsError(Exception):
    pass


class FileLoaderError(EvalUtilsError):
    pass


class ValidationError(EvalUtilsError):
    pass


class ConfigurationError(EvalUtilsError):
    pass
