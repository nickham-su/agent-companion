"""用 Darwin 的扩展 ACL 接口检查已经打开的归档目录或文件。"""

import ctypes
from errno import EINVAL, ENOENT
from functools import lru_cache
import sys


# Apple Libc include/sys/acl.h：acl_get_fd 等价于 acl_get_fd_np(fd, ACL_TYPE_EXTENDED)。
_ACL_FIRST_ENTRY = 0
_ACL_NEXT_ENTRY = -1
_ACL_EXTENDED_DENY = 2


@lru_cache(maxsize=1)
def _acl_api():
    try:
        api = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        api.acl_get_fd.argtypes, api.acl_get_fd.restype = [integer], pointer
        api.acl_valid.argtypes, api.acl_valid.restype = [pointer], integer
        api.acl_get_entry.argtypes, api.acl_get_entry.restype = [pointer, integer, ctypes.POINTER(pointer)], integer
        api.acl_get_tag_type.argtypes, api.acl_get_tag_type.restype = [pointer, ctypes.POINTER(integer)], integer
        api.acl_free.argtypes, api.acl_free.restype = [pointer], integer
        return api
    except (AttributeError, OSError):
        raise OSError("Darwin ACL interface unavailable") from None


def require_no_acl_grants(fd: int) -> None:
    if sys.platform != "darwin":
        return
    api = _acl_api()
    ctypes.set_errno(0)
    acl = api.acl_get_fd(fd)
    if not acl:
        # Libc acl_file.c + filesec.c: 打开的 FD 没有 FILESEC_ACL 时返回 NULL / ENOENT。
        # ENOTSUP 等错误不是「没有 ACL」的证据，按不可检处理。
        if ctypes.get_errno() == ENOENT:
            return
        raise OSError("Darwin ACL query failed")
    try:
        if api.acl_valid(acl) != 0:
            raise OSError("invalid Darwin ACL")
        entry = ctypes.c_void_p()
        tag = ctypes.c_int()
        selector = _ACL_FIRST_ENTRY
        while True:
            ctypes.set_errno(0)
            result = api.acl_get_entry(acl, selector, ctypes.byref(entry))
            # Apple acl_entry.c: 已验证的 ACL 用 FIRST/NEXT 遍历，空表或到末尾是 -1/EINVAL。
            if result == -1 and ctypes.get_errno() == EINVAL:
                return
            if result != 0 or not entry.value or api.acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                raise OSError("Darwin ACL entry query failed")
            # 不计算授予对象或权限掩码；任何 allow（含仅继承）都保守拒绝。
            if tag.value != _ACL_EXTENDED_DENY:
                raise OSError("Darwin ACL may grant archive access")
            selector = _ACL_NEXT_ENTRY
    finally:
        api.acl_free(acl)
