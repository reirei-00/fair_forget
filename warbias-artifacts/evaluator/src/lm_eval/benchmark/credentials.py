"""Read explicitly selected local credentials without writing plaintext files."""

import ctypes
import os
import sys
from contextlib import contextmanager


def mac_keychain_secret(service, account):
    """Read one named generic-password item; never print its data."""
    if sys.platform != "darwin":
        raise ValueError(
            "macOS Keychain is unavailable; set the credential environment variable"
        )
    if not all(isinstance(value, str) and value for value in (service, account)):
        raise ValueError("Keychain service and account must be explicit")
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    find = security.SecKeychainFindGenericPassword
    find.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    find.restype = ctypes.c_int32
    free = security.SecKeychainItemFreeContent
    free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    free.restype = ctypes.c_int32
    service_bytes, account_bytes = service.encode(), account.encode()
    length, pointer = ctypes.c_uint32(), ctypes.c_void_p()
    status = find(
        None,
        len(service_bytes),
        service_bytes,
        len(account_bytes),
        account_bytes,
        ctypes.byref(length),
        ctypes.byref(pointer),
        None,
    )
    if status != 0:
        raise ValueError(
            f"Selected Keychain credential is unavailable (status {status})"
        )
    try:
        secret = ctypes.string_at(pointer, length.value).decode()
    finally:
        free(None, pointer)
    if not secret:
        raise ValueError("Selected Keychain credential is empty")
    return secret


@contextmanager
def credential_environment(name, selection=None):
    """Expose an explicit Keychain item only during an authorized API operation."""
    if os.environ.get(name):
        yield
        return
    if selection is None:
        raise ValueError(f"Set {name} in the execution environment")
    if not isinstance(selection, dict) or set(selection) != {"service", "account"}:
        raise ValueError("Use an explicit Keychain service/account selection")
    previous = os.environ.get(name)
    os.environ[name] = mac_keychain_secret(selection["service"], selection["account"])
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
