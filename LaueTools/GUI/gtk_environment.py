"""
Environment of the GTK file dialogs of the LaueTools GUIs (Linux), set before wx creates its first window.

'Recent' place of the GTK file chooser: GTK shows it only if the location recent:/// is supported, i.e. if GLib finds
the GVfs module (libgvfsdbus.so) and a D-Bus session bus (to reach the GVfs daemon). In a conda environment, GLib only
looks for its modules in the environment (<env>/lib/gio/modules, without GVfs), so 'Recent' is shown or not depending
on how the session was started. setup() adds the GIO modules folder of the system (GIO_EXTRA_MODULES) and starts a
D-Bus session bus if there is none. LAUETOOLS_NO_GTK_SETUP=1: nothing done.
"""
import os
import shutil
import subprocess
import sys

SYSTEM_GIO_MODULES = ('/usr/lib/x86_64-linux-gnu/gio/modules', '/usr/lib64/gio/modules', '/usr/lib/gio/modules',
                      '/usr/lib/aarch64-linux-gnu/gio/modules')


def _system_gvfs_modules():
    for folder in SYSTEM_GIO_MODULES:
        if os.path.isfile(os.path.join(folder, 'libgvfsdbus.so')):
            return folder
    return None


def _start_dbus_session():
    """start a D-Bus session bus (system dbus-launch: its configuration knows the GVfs services) and set
    DBUS_SESSION_BUS_ADDRESS. False if not possible"""
    launcher = '/usr/bin/dbus-launch' if os.path.exists('/usr/bin/dbus-launch') else shutil.which('dbus-launch')
    if launcher is None:
        return False
    try:
        out = subprocess.run([launcher, '--sh-syntax'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():   # DBUS_SESSION_BUS_ADDRESS='unix:...'; export DBUS_SESSION_BUS_ADDRESS;
        if '=' in line and line.startswith('DBUS_'):
            key, value = line.split(';')[0].split('=', 1)
            os.environ[key] = value.strip("'\"")
    return 'DBUS_SESSION_BUS_ADDRESS' in os.environ


def setup(verbose=False):
    """set the environment of the GTK file dialogs ('Recent' place). To be called before the first wx window"""
    if not sys.platform.startswith('linux') or os.environ.get('LAUETOOLS_NO_GTK_SETUP'):
        return
    modules = _system_gvfs_modules()
    if modules is not None:
        extra = [p for p in os.environ.get('GIO_EXTRA_MODULES', '').split(':') if p]
        if modules not in extra:
            os.environ['GIO_EXTRA_MODULES'] = ':'.join(extra + [modules])
    if not os.environ.get('DBUS_SESSION_BUS_ADDRESS'):
        bus = f"/run/user/{os.getuid()}/bus"
        if os.path.exists(bus):   # session bus of the user (systemd)
            os.environ['DBUS_SESSION_BUS_ADDRESS'] = f'unix:path={bus}'
        else:
            _start_dbus_session()
    if verbose:
        print('GIO_EXTRA_MODULES', os.environ.get('GIO_EXTRA_MODULES'),
              'DBUS_SESSION_BUS_ADDRESS', os.environ.get('DBUS_SESSION_BUS_ADDRESS'))
