import os as _os
import sys as _sys


def _disable_dbus_in_remote_session():
    """GTK (wxPython GUIs) asks the D-Bus session for the desktop portal when it starts. In a ssh
    session (with X11 forwarding) on a machine where the portal service cannot start without a
    graphical session (e.g. lbm32gpu1), GTK waits for the D-Bus timeout (25 s) at each GUI launch.
    LaueTools does not need D-Bus: it is disabled for this process (and its subprocesses) in such
    remote sessions. Set LAUETOOLS_KEEP_DBUS=1 to keep it."""
    env = _os.environ
    if (_sys.platform.startswith('linux')
            and (env.get('SSH_CONNECTION') or env.get('SSH_CLIENT'))
            and env.get('XDG_SESSION_TYPE') not in ('x11', 'wayland')
            and not env.get('LAUETOOLS_KEEP_DBUS')):
        env['DBUS_SESSION_BUS_ADDRESS'] = 'disabled:'


_disable_dbus_in_remote_session()
