# environment of the GTK file dialogs ('Recent' place of the file chooser), before wx creates its first window
from LaueTools.GUI import gtk_environment as _gtk_environment

_gtk_environment.setup()
