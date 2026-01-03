"""
LLM utility functions for robust invocation with retry logic and timeout handling.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time
from config import LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES


def invoke_with_retry(chain, input_data, operation_name: str = "LLM call", timeout_seconds: int = None, max_retries: int = None):
    """
    Invoke an LLM chain with timeout and retry logic.

    Args:
        chain: The LangChain chain/agent to invoke
        input_data: Input to pass to the chain
        operation_name: Human-readable name for logging
        timeout_seconds: Timeout for each LLM call (default: uses LLM_TIMEOUT_SECONDS from config)
        max_retries: Maximum retry attempts (default: uses LLM_MAX_RETRIES from config)

    Returns:
        The response from the chain

    Raises:
        Exception: If all retries fail
    """
    # Use config defaults if not provided
    if timeout_seconds is None:
        timeout_seconds = LLM_TIMEOUT_SECONDS
    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    logger = logging.getLogger(__name__)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[{operation_name}] Attempt {attempt}/{max_retries} (timeout: {timeout_seconds}s)")

            # Use ThreadPoolExecutor to run with timeout
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(chain.invoke, input_data)
                result = future.result(timeout=timeout_seconds)
                logger.info(f"[{operation_name}] Success on attempt {attempt}")
                executor.shutdown(wait=False)
                return result
            except FuturesTimeoutError:
                logger.warning(f"[{operation_name}] Timeout after {timeout_seconds}s on attempt {attempt}")
                # Cancel the future and shutdown executor without waiting
                future.cancel()
                executor.shutdown(wait=False)

                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    logger.info(f"[{operation_name}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    logger.info(f"[{operation_name}] Retrying now...")
                else:
                    raise TimeoutError(f"{operation_name} timed out after {max_retries} attempts")
            except Exception as e:
                logger.error(f"[{operation_name}] Error on attempt {attempt}: {e}")
                executor.shutdown(wait=False)
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"[{operation_name}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    logger.info(f"[{operation_name}] Retrying now...")
                else:
                    raise

        except TimeoutError:
            raise  # Re-raise timeout error

    raise Exception(f"{operation_name} failed after {max_retries} attempts")
