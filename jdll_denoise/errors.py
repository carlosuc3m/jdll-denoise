"""Public exception hierarchy for jdll-denoise."""


class DenoisingError(RuntimeError):
    """Base class for errors safe to present to an end user."""


class ConfigurationError(DenoisingError):
    """The requested configuration is invalid."""


class InputFormatError(DenoisingError):
    """The input array or axes are invalid."""


class MethodUnavailableError(DenoisingError):
    """A method's optional implementation is not installed."""


class DeviceUnavailableError(DenoisingError):
    """The requested compute device cannot be used."""


class DenoisingCancelled(DenoisingError):
    """Cooperative cancellation was requested."""


class OutOfMemoryDenoisingError(DenoisingError):
    """Denoising ran out of memory after safe recovery attempts."""


class ThirdPartyBackendError(DenoisingError):
    """An optional third-party method failed."""
