import os as _os
import sys as _sys


def _dbus_session_responsive(timeout_ms: int = 2000) -> bool:
    """True if the D-Bus session bus and the desktop portal answer within timeout_ms (dbus-send ping).
    No dbus-send: True (nothing known, D-Bus kept)"""
    import shutil
    import subprocess
    sender = shutil.which('dbus-send')
    if sender is None:
        return True
    try:
        res = subprocess.run([sender, '--session', '--print-reply', f'--reply-timeout={timeout_ms}',
                              '--dest=org.freedesktop.portal.Desktop', '/', 'org.freedesktop.DBus.Peer.Ping'],
                             capture_output=True, timeout=timeout_ms / 1000 + 2)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


def _check_dbus_in_remote_session():
    """GTK (wxPython GUIs) asks the D-Bus session for the desktop portal when it starts. In a ssh session (with X11
    forwarding) where the portal service cannot start (no graphical session), GTK waits for the D-Bus timeout (25 s)
    at each GUI launch. In such remote sessions, D-Bus is disabled for this process (and its subprocesses) only if the
    portal does not answer a ping within 2 s; otherwise it is kept: it is needed by the 'Recent' place of the GTK file
    dialogs (GVfs). Done once (LAUETOOLS_DBUS_CHECKED, inherited by subprocesses).
    LAUETOOLS_KEEP_DBUS=1: D-Bus always kept; LAUETOOLS_DISABLE_DBUS=1: always disabled in remote sessions."""
    env = _os.environ
    if (not _sys.platform.startswith('linux') or env.get('LAUETOOLS_DBUS_CHECKED') or env.get('LAUETOOLS_KEEP_DBUS')
            or not (env.get('SSH_CONNECTION') or env.get('SSH_CLIENT'))
            or env.get('XDG_SESSION_TYPE') in ('x11', 'wayland')):
        return
    env['LAUETOOLS_DBUS_CHECKED'] = '1'
    if env.get('LAUETOOLS_DISABLE_DBUS') or not env.get('DBUS_SESSION_BUS_ADDRESS') \
            or not _dbus_session_responsive():
        env['DBUS_SESSION_BUS_ADDRESS'] = 'disabled:'


_check_dbus_in_remote_session()
