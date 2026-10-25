"""Exceptions whose messages are safe to show users, plus mapping of raw LLM errors to them."""
from __future__ import annotations


class DocuMindError(Exception):
    """Base class. str(exc) is always a friendly message with no technical details."""


class InvalidFileError(DocuMindError):
    pass


class EmptyDocumentError(DocuMindError):
    pass


class ConfigurationError(DocuMindError):
    pass


class VectorStoreError(DocuMindError):
    pass


class EmbeddingError(DocuMindError):
    pass


class LLMError(DocuMindError):
    pass


def classify_llm_error(exc: Exception) -> DocuMindError:
    """Translate a provider exception into a friendly error (matching on text keeps this SDK-agnostic)."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(s in text for s in ("api key not valid", "api_key_invalid", "invalid api key", "unauthenticated", "permissiondenied", "permission_denied")):
        return ConfigurationError("The Gemini API key was rejected. Check that GOOGLE_API_KEY is correct and has access to the Gemini API.")
    if any(s in text for s in ("429", "resourceexhausted", "resource_exhausted", "quota", "rate limit")):
        return LLMError("The Gemini API rate limit or quota was reached. Please wait a minute and try again.")
    if "404" in text or ("model" in text and "not found" in text):
        return ConfigurationError("The configured Gemini model is not available. Check GEMINI_MODEL in your settings.")
    if any(s in text for s in ("connect", "timeout", "timed out", "unavailable", "503", "network", "dns")):
        return LLMError("Could not reach the Gemini API. Check your internet connection and try again.")
    return LLMError("The language model could not complete the request. Please try again.")
