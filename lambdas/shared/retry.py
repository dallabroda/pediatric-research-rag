"""
Retry logic with exponential backoff for AWS API calls.

Provides decorators and utilities for handling transient failures
in Bedrock, S3, and external API calls.
"""
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Type

from botocore.exceptions import ClientError

# Optional requests import for Lambda compatibility
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# Default retryable exceptions
_base_exceptions: list[Type[Exception]] = [ConnectionError, TimeoutError]
if HAS_REQUESTS and requests is not None:
    _base_exceptions.extend([requests.exceptions.Timeout, requests.exceptions.ConnectionError])
RETRYABLE_EXCEPTIONS: tuple[Type[Exception], ...] = tuple(_base_exceptions)


def is_throttling_error(exception: Exception) -> bool:
    """
    Check if an exception is a throttling/rate limit error.

    Args:
        exception: The exception to check

    Returns:
        True if this is a throttling error that should be retried
    """
    # Check Bedrock/AWS throttling
    if isinstance(exception, ClientError):
        error_code = exception.response.get("Error", {}).get("Code", "")
        return error_code in (
            "ThrottlingException",
            "TooManyRequestsException",
            "ProvisionedThroughputExceededException",
            "ServiceUnavailable",
        )

    # Check exception name for throttling patterns
    exc_name = type(exception).__name__
    if "Throttl" in exc_name or "RateLimit" in exc_name:
        return True

    # Check HTTP status codes for requests exceptions
    if HAS_REQUESTS and requests is not None:
        if isinstance(exception, requests.exceptions.HTTPError):
            response = getattr(exception, "response", None)
            if response is not None:
                return response.status_code in (429, 503, 502)

    return False


def retry_with_backoff(
    func: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Automatically retries on:
    - Throttling errors from AWS services
    - Connection errors
    - Timeout errors
    - Configurable additional exception types

    Args:
        func: Function to wrap (if called without arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff (delay doubles by default)
        jitter: Add random jitter to prevent thundering herd
        retry_exceptions: Additional exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_bedrock_api(text):
            return bedrock_client.invoke_model(...)

        # Or with default settings
        @retry_with_backoff
        def simple_api_call():
            return requests.get(...)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            exceptions_to_catch = RETRYABLE_EXCEPTIONS
            if retry_exceptions:
                exceptions_to_catch = exceptions_to_catch + retry_exceptions

            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    should_retry = False

                    # Check if it's a throttling error
                    if is_throttling_error(e):
                        should_retry = True
                        error_type = "throttling"
                    # Check if it's a retryable exception type
                    elif isinstance(e, exceptions_to_catch):
                        should_retry = True
                        error_type = "connection/timeout"
                    else:
                        # Non-retryable error, raise immediately
                        raise

                    if not should_retry or attempt >= max_retries:
                        logger.error(
                            f"Function {fn.__name__} failed after {attempt + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Function {fn.__name__} {error_type} error on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    time.sleep(delay)

            # This should not be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    # Handle both @retry_with_backoff and @retry_with_backoff() syntax
    if func is not None:
        return decorator(func)
    return decorator


# Only define RetryableAPIClient if requests is available
if HAS_REQUESTS and requests is not None:
    class RetryableAPIClient:
        """
        Base class for API clients with built-in retry logic.

        Provides a request method that automatically retries on failures.
        Requires the requests library to be installed.
        """

        def __init__(
            self,
            max_retries: int = 3,
            base_delay: float = 1.0,
            max_delay: float = 60.0,
            timeout: int = 30,
        ):
            """
            Initialize the retryable client.

            Args:
                max_retries: Maximum retry attempts
                base_delay: Initial backoff delay in seconds
                max_delay: Maximum backoff delay in seconds
                timeout: Request timeout in seconds
            """
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            self.timeout = timeout
            self._session: Optional[requests.Session] = None

        @property
        def session(self) -> requests.Session:
            """Lazy-load requests session."""
            if self._session is None:
                self._session = requests.Session()
            return self._session

        def request(
            self,
            method: str,
            url: str,
            **kwargs: Any,
        ) -> requests.Response:
            """
            Make an HTTP request with automatic retry.

            Args:
                method: HTTP method (GET, POST, etc.)
                url: Request URL
                **kwargs: Additional arguments passed to requests

            Returns:
                Response object

            Raises:
                requests.exceptions.RequestException: After all retries exhausted
            """
            kwargs.setdefault("timeout", self.timeout)

            @retry_with_backoff(
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
            )
            def _make_request() -> requests.Response:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            return _make_request()

        def get(self, url: str, **kwargs: Any) -> requests.Response:
            """Make a GET request with retry."""
            return self.request("GET", url, **kwargs)

        def post(self, url: str, **kwargs: Any) -> requests.Response:
            """Make a POST request with retry."""
            return self.request("POST", url, **kwargs)
