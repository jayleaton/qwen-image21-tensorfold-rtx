"""Windows kill-on-close job for the API's own backend, nested in tray jobs."""
import ctypes
from ctypes import wintypes


class WindowsJob:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION is 144 bytes on Windows x64;
        # LimitFlags is the DWORD at offset 16 in BASIC_LIMIT_INFORMATION.
        info = ctypes.create_string_buffer(144)
        ctypes.c_uint32.from_buffer(info, 16).value = 0x2000
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, info, len(info)):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, proc):
        if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(proc._handle))):
            proc.terminate()
            proc.wait()
            raise ctypes.WinError(ctypes.get_last_error())
        # The child was created suspended, so no descendant can escape
        # before adoption. Resume its initial thread only after assignment.
        class ThreadEntry(ctypes.Structure):
            _fields_ = [(name, wintypes.DWORD) for name in
                ("size", "usage", "thread_id", "owner_pid", "base_priority", "delta_priority", "flags")]
        k = self.kernel
        k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        k.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        k.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenThread.restype = wintypes.HANDLE
        k.ResumeThread.argtypes = [wintypes.HANDLE]
        k.ResumeThread.restype = wintypes.DWORD
        snapshot = k.CreateToolhelp32Snapshot(4, 0)
        entry = ThreadEntry()
        entry.size = ctypes.sizeof(entry)
        resumed = False
        try:
            ok = k.Thread32First(snapshot, ctypes.byref(entry))
            while ok:
                if entry.owner_pid == proc.pid:
                    thread = k.OpenThread(2, False, entry.thread_id)
                    if thread:
                        try:
                            resumed = k.ResumeThread(thread) != 0xFFFFFFFF
                        finally:
                            k.CloseHandle(thread)
                ok = k.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            k.CloseHandle(snapshot)
        if not resumed:
            proc.terminate()
            proc.wait()
            raise RuntimeError("Could not resume the owned backend process")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
