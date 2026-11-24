"""Linux 本地组状态探测；由 Bash 执行器按独立时限运行。"""

import os
import sys


def live_members(group: int) -> bool | None:
    found_zombie = False
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            try:
                # 先问内核组号，不读取无关 PID 的 stat（可能没有权限）。
                if os.getpgid(int(entry.name)) != group:
                    continue
                with open(f"{entry.path}/stat", "rb") as stat:
                    # comm 可含空格和括号，最后一个 ')' 才是字段结束。
                    state, _, pgid, sid, _ = stat.read().rsplit(b") ", 1)[-1].split(maxsplit=4)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(pgid) == group and int(sid) == group:
                if state not in (b"Z", b"X", b"x"):
                    return True
                found_zombie = True
    return False if found_zombie else None


if __name__ == "__main__":
    status = live_members(int(sys.argv[1]))
    print({True: "live", False: "zombies", None: "unknown"}[status])
