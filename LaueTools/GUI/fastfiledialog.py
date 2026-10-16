"""
Fast dialog to open a file in folders with many files (e.g. 30000 images or peak lists on GPFS)

The GTK file dialog reads the properties (type, size, date ...) of every file of a folder before it can be used,
which takes minutes in such folders. FastFileDialog only reads the names (os.scandir, once per folder):
    - type (or paste) the path and press Enter; Tab or the list completes the names
    - the list shows the names of the folder starting with the typed text (folders first)
    - recent files of the LaueTools GUIs (~/.config/LaueTools/recent_files.json)
    - bookmarked folders, kept from one session to the next (~/.config/LaueTools/bookmarks.json): 'Bookmark this
      folder', 'Remove bookmark'; a click opens the folder
    - 'Browse...' opens the system (GTK) dialog, with its 'Recent' place

FastFileDialog has the methods of wx.FileDialog used by the GUIs (ShowModal, GetPath, GetPaths, GetFilename,
GetDirectory, SetMessage, SetDirectory, SetFilename, SetPath, SetWildcard, Get/SetFilterIndex, Destroy): replace
wx.FileDialog(...) by FastFileDialog(...) for dialogs opening a file. Save and multiple selection dialogs
(wx.FD_SAVE, wx.FD_MULTIPLE), or LAUETOOLS_SYSTEM_FILEDIALOG=1: the system dialog is used.
"""
import fnmatch
import json
import os
import re
import time

import wx

RECENT_FILE = os.path.join(os.path.expanduser('~'), '.config', 'LaueTools', 'recent_files.json')
BOOKMARKS_FILE = os.path.join(os.path.expanduser('~'), '.config', 'LaueTools', 'bookmarks.json')
NB_RECENT = 30
_FOLDER_CACHE = {}        # folder -> (time, [(name, is_dir)]), names read once per folder (refreshed after 60 s)


def parse_wildcard(wildcard: str) -> list:
    """'images (*.tif;*.h5)|*.tif;*.h5|All files(*)|*' -> [('images (*.tif;*.h5)', ['*.tif', '*.h5']), ...]"""
    parts = [p for p in str(wildcard or '*').split('|')]
    if len(parts) == 1:
        return [(parts[0], [p.strip() for p in parts[0].split(';') if p.strip()] or ['*'])]
    return [(parts[k], [p.strip() for p in parts[k + 1].split(';') if p.strip()] or ['*'])
            for k in range(0, len(parts) - 1, 2)]


def _natural_key(name: str):
    """sort key with the numbers in their numerical order: img_9999 before img_10000"""
    return [(0, int(part), '') if part.isdigit() else (1, 0, part.lower()) for part in re.split(r'(\d+)', name)]


def folder_names(folder: str, refresh: bool = False) -> list:
    """[(name, is_dir)] of a folder, sorted (folders first, numbers in their numerical order), read once
    (os.scandir: names only)"""
    folder = os.path.abspath(folder)
    cached = _FOLDER_CACHE.get(folder)
    if cached is not None and not refresh and time.time() - cached[0] < 60:
        return cached[1]
    names = []
    try:
        with os.scandir(folder) as it:
            for entry in it:
                try:
                    isdir = entry.is_dir()
                except OSError:
                    isdir = False
                names.append((entry.name, isdir))
    except OSError:
        return []
    names.sort(key=lambda e: (not e[1], _natural_key(e[0])))
    _FOLDER_CACHE[folder] = (time.time(), names)
    return names


def read_recent() -> list:
    try:
        with open(RECENT_FILE) as f:
            return [p for p in json.load(f) if isinstance(p, str)]
    except (OSError, ValueError):
        return []


def add_recent(path: str):
    """add a file to the recent files of the LaueTools GUIs (and to the GTK recent files)"""
    path = os.path.abspath(path)
    recent = [path] + [p for p in read_recent() if p != path]
    try:
        os.makedirs(os.path.dirname(RECENT_FILE), exist_ok=True)
        with open(RECENT_FILE, 'w') as f:
            json.dump(recent[:NB_RECENT], f, indent=0)
    except OSError:
        pass


def read_bookmarks() -> list:
    """bookmarked folders (kept from one session to the next)"""
    try:
        with open(BOOKMARKS_FILE) as f:
            return [p for p in json.load(f) if isinstance(p, str)]
    except (OSError, ValueError):
        return []


def write_bookmarks(folders: list):
    try:
        os.makedirs(os.path.dirname(BOOKMARKS_FILE), exist_ok=True)
        with open(BOOKMARKS_FILE, 'w') as f:
            json.dump(list(folders), f, indent=0)
    except OSError as err:
        wx.MessageBox(f'bookmarks could not be saved in {BOOKMARKS_FILE}:\n{err}', 'Bookmarks', wx.ICON_ERROR)


def add_bookmark(folder: str) -> list:
    folder = os.path.abspath(os.path.expanduser(folder)).rstrip(os.sep) or os.sep
    bookmarks = read_bookmarks()
    if folder not in bookmarks:
        bookmarks.append(folder)
        write_bookmarks(bookmarks)
    return bookmarks


def remove_bookmark(folder: str) -> list:
    bookmarks = [b for b in read_bookmarks() if b != folder]
    write_bookmarks(bookmarks)
    return bookmarks


class FastFileDialog:
    """dialog to open a file, with the API of wx.FileDialog (see module docstring)"""

    def __init__(self, parent, message='Choose a file', defaultDir='', defaultFile='', wildcard='*',
                 style=wx.FD_OPEN, pos=wx.DefaultPosition, size=wx.DefaultSize, name='filedlg'):
        self.parent = parent
        self.message, self.wildcard, self.style = message, wildcard or '*', style
        self.directory = os.path.abspath(os.path.expanduser(defaultDir)) if defaultDir else os.getcwd()
        self.filename = defaultFile or ''
        self.filterindex = 0
        self.path = ''
        self._native = None
        if style & (wx.FD_SAVE | wx.FD_MULTIPLE) or os.environ.get('LAUETOOLS_SYSTEM_FILEDIALOG'):
            self._native = wx.FileDialog(parent, message, defaultDir, defaultFile, wildcard, style, pos, size, name)

    # ------------------------------------------------------------------ wx.FileDialog API
    def __getattr__(self, attr):   # other methods: those of the system dialog
        native = self.__dict__.get('_native')
        if native is not None:
            return getattr(native, attr)
        raise AttributeError(attr)

    def ShowModal(self):
        if self._native is not None:
            res = self._native.ShowModal()
            if res == wx.ID_OK and not self.style & wx.FD_SAVE:
                for p in self._native.GetPaths():
                    add_recent(p)
            return res
        dlg = _FastFileDialogWindow(self)
        res = dlg.ShowModal()
        if res == wx.ID_OK:
            self.path = dlg.path
            self.directory, self.filename = os.path.split(self.path)
            self.filterindex = dlg.filter.GetSelection()
            add_recent(self.path)
            if self.style & wx.FD_CHANGE_DIR:
                os.chdir(self.directory)
        dlg.Destroy()
        return res

    def GetPath(self):
        return self._native.GetPath() if self._native is not None else self.path

    def GetPaths(self):
        return self._native.GetPaths() if self._native is not None else ([self.path] if self.path else [])

    def GetFilename(self):
        return self._native.GetFilename() if self._native is not None else self.filename

    def GetDirectory(self):
        return self._native.GetDirectory() if self._native is not None else self.directory

    def SetMessage(self, message):
        self.message = message
        if self._native is not None:
            self._native.SetMessage(message)

    def SetDirectory(self, directory):
        self.directory = os.path.abspath(os.path.expanduser(directory))
        if self._native is not None:
            self._native.SetDirectory(directory)

    def SetFilename(self, filename):
        self.filename = filename
        if self._native is not None:
            self._native.SetFilename(filename)

    def SetPath(self, path):
        self.directory, self.filename = os.path.split(os.path.abspath(os.path.expanduser(path)))
        if self._native is not None:
            self._native.SetPath(path)

    def SetWildcard(self, wildcard):
        self.wildcard = wildcard
        if self._native is not None:
            self._native.SetWildcard(wildcard)

    def GetFilterIndex(self):
        return self._native.GetFilterIndex() if self._native is not None else self.filterindex

    def SetFilterIndex(self, index):
        self.filterindex = index
        if self._native is not None:
            self._native.SetFilterIndex(index)

    def Destroy(self):
        if self._native is not None:
            self._native.Destroy()
            self._native = None


class _NamesList(wx.ListCtrl):
    """virtual list of the names of a folder: only the visible lines are drawn (any nb of names, e.g. 60000)"""

    def __init__(self, parent):
        super().__init__(parent, -1, style=wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_SINGLE_SEL | wx.LC_NO_HEADER)
        self.InsertColumn(0, 'name')
        self.items = []
        self.Bind(wx.EVT_SIZE, self._on_size)

    def _on_size(self, event):
        self.SetColumnWidth(0, max(self.GetClientSize()[0] - 4, 100))
        event.Skip()

    def set_items(self, items):
        self.items = items
        self.SetItemCount(len(items))
        if items:
            self.EnsureVisible(0)
        self.Refresh()

    def OnGetItemText(self, item, column):
        return self.items[item] if 0 <= item < len(self.items) else ''

    def GetSelection(self):
        return self.GetFirstSelected()

    def SetSelection(self, k):
        self.Select(k)
        self.Focus(k)

    def GetCount(self):
        return len(self.items)


class _FastFileDialogWindow(wx.Dialog):
    """window of FastFileDialog: path field, names of the folder, recent files, Browse..."""

    def __init__(self, spec: FastFileDialog):
        super().__init__(spec.parent, title=spec.message or 'Open a file', size=(1040, 660),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.spec, self.path = spec, ''
        self.filters = parse_wildcard(spec.wildcard)

        panel = self
        help_ = wx.StaticText(panel, -1, 'Type or paste the path and press Enter (Tab: completion). Double click: open '
                                         'a file or a folder, one click on ../ or \u2191: parent folder. '
                                         'Bookmarks (left): kept from one session to the next. '
                                         'Browse...: system dialog')
        self.text = wx.TextCtrl(panel, -1, os.path.join(spec.directory, spec.filename), style=wx.TE_PROCESS_ENTER)
        self.filter = wx.Choice(panel, -1, choices=[f[0] for f in self.filters])
        self.filter.SetSelection(min(spec.filterindex, len(self.filters) - 1))
        self.info = wx.StaticText(panel, -1, '')
        self.names = _NamesList(panel)
        recentlabel = wx.StaticText(panel, -1, 'Recent files (double click to open):')
        self.recent = wx.ListBox(panel, -1, choices=read_recent(), style=wx.LB_SINGLE, size=(-1, 130))
        bookmarklabel = wx.StaticText(panel, -1, 'Bookmarked folders:')
        self.bookmarks = wx.ListBox(panel, -1, style=wx.LB_SINGLE, size=(230, -1))
        self.bookmarks.SetToolTip('click: open the folder (bookmarks kept from one session to the next)')
        addbmbtn = wx.Button(panel, -1, 'Bookmark this folder')
        delbmbtn = wx.Button(panel, -1, 'Remove bookmark')
        self._refresh_bookmarks()
        upbtn = wx.Button(panel, -1, 'Parent folder')
        browsebtn = wx.Button(panel, -1, 'Browse...')
        openbtn = wx.Button(panel, wx.ID_OK, 'Open')
        cancelbtn = wx.Button(panel, wx.ID_CANCEL, 'Cancel')
        openbtn.SetDefault()

        upbtn_arrow = wx.Button(panel, -1, '\u2191', size=(36, -1))   # (up arrow)
        upbtn_arrow.SetToolTip('parent folder')
        upbtn_arrow.Bind(wx.EVT_BUTTON, self.on_parent)
        top = wx.BoxSizer(wx.HORIZONTAL)
        top.Add(upbtn_arrow, 0, wx.RIGHT, 4)
        top.Add(self.text, 1, wx.EXPAND | wx.RIGHT, 5)
        top.Add(self.filter, 0)
        btns = wx.BoxSizer(wx.HORIZONTAL)
        btns.Add(upbtn, 0, wx.RIGHT, 5)
        btns.Add(browsebtn, 0)
        btns.AddStretchSpacer()
        btns.Add(cancelbtn, 0, wx.RIGHT, 5)
        btns.Add(openbtn, 0)
        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(top, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        main.Add(self.info, 0, wx.ALL, 6)
        main.Add(self.names, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        main.Add(recentlabel, 0, wx.ALL, 6)
        main.Add(self.recent, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        side = wx.BoxSizer(wx.VERTICAL)   # bookmarks (left, as the places of the system dialog)
        side.Add(bookmarklabel, 0, wx.BOTTOM, 4)
        side.Add(self.bookmarks, 1, wx.EXPAND)
        side.Add(addbmbtn, 0, wx.EXPAND | wx.TOP, 4)
        side.Add(delbmbtn, 0, wx.EXPAND | wx.TOP, 4)
        middle = wx.BoxSizer(wx.HORIZONTAL)
        middle.Add(side, 0, wx.EXPAND | wx.LEFT, 6)
        middle.Add(main, 1, wx.EXPAND)
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(help_, 0, wx.ALL, 6)
        vbox.Add(middle, 1, wx.EXPAND)
        vbox.Add(btns, 0, wx.EXPAND | wx.ALL, 6)
        self.SetSizer(vbox)

        self.text.Bind(wx.EVT_TEXT, self.on_text)
        self.text.Bind(wx.EVT_TEXT_ENTER, self.on_enter)
        self.text.Bind(wx.EVT_CHAR_HOOK, self.on_key)
        self.filter.Bind(wx.EVT_CHOICE, self.on_text)
        self.names.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_name_selected)
        self.names.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_name_dclick)   # double click or Enter
        self.recent.Bind(wx.EVT_LISTBOX_DCLICK, self.on_recent_dclick)
        self.bookmarks.Bind(wx.EVT_LISTBOX, self.on_bookmark)
        addbmbtn.Bind(wx.EVT_BUTTON, self.on_add_bookmark)
        delbmbtn.Bind(wx.EVT_BUTTON, self.on_remove_bookmark)
        upbtn.Bind(wx.EVT_BUTTON, self.on_parent)
        browsebtn.Bind(wx.EVT_BUTTON, self.on_browse)
        openbtn.Bind(wx.EVT_BUTTON, self.on_enter)

        self.text.SetFocus()
        self.text.SetInsertionPointEnd()
        self._shown = []
        self.on_text(None)

    # ---------------------------------------------------------------- helpers
    def _split(self):
        """(folder, typed beginning of the name) of the text"""
        text = os.path.expanduser(self.text.GetValue().strip())
        if not text:
            return self.spec.directory, ''
        if text.endswith(os.sep) or os.path.isdir(text) and not os.path.basename(text).startswith('.'):
            return (text if os.path.isdir(text) else os.path.dirname(text)), ''
        return os.path.dirname(text) or self.spec.directory, os.path.basename(text)

    def _matches(self, folder, prefix):
        patterns = self.filters[self.filter.GetSelection()][1] if self.filters else ['*']
        low = prefix.lower()
        out = []
        for name, isdir in folder_names(folder):
            if name.startswith('.') and not prefix.startswith('.'):
                continue
            if not name.lower().startswith(low):
                continue
            if isdir or any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns):
                out.append((name, isdir))
        return out

    def _accept(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if self.spec.style & wx.FD_FILE_MUST_EXIST and not os.path.isfile(path):
            wx.MessageBox(f'{path}\ndoes not exist', 'Open a file', wx.ICON_ERROR)
            return
        self.path = path
        self.EndModal(wx.ID_OK)

    def _set_text(self, value):
        self.text.ChangeValue(value)
        self.text.SetInsertionPointEnd()
        self.on_text(None)

    # ---------------------------------------------------------------- events
    def on_text(self, _):
        folder, prefix = self._split()
        if not os.path.isdir(folder):
            self.names.set_items([])
            self.info.SetLabel(f'folder not found: {folder}')
            return
        wx.BeginBusyCursor()
        try:
            matches = self._matches(folder, prefix)
        finally:
            wx.EndBusyCursor()
        self._shown = matches
        parent = os.path.dirname(os.path.abspath(folder).rstrip(os.sep))
        if not prefix and parent and parent != os.path.abspath(folder):   # '..': parent folder (one click)
            self._shown = [('..', True)] + self._shown
        self.names.set_items([name + (os.sep if isdir else '') for name, isdir in self._shown])
        more = ''
        self.info.SetLabel(f'{folder}: {len(matches)} matching name(s){more}'
                           + ('   (../ or \u2191: parent folder)' if self._shown and self._shown[0][0] == '..' else ''))
        self._folder = folder

    def on_key(self, event):
        if event.GetKeyCode() == wx.WXK_TAB:   # completion: common beginning of the matching names
            folder, prefix = self._split()
            names = [n for n, _ in self._matches(folder, prefix)] if os.path.isdir(folder) else []
            if names:
                common = os.path.commonprefix(names)
                if len(names) == 1 and os.path.isdir(os.path.join(folder, common)):
                    common += os.sep
                self._set_text(os.path.join(folder, common))
            return
        if event.GetKeyCode() == wx.WXK_DOWN and self.names.GetCount():
            self.names.SetSelection(0)
            self.names.SetFocus()
            return
        event.Skip()

    def on_enter(self, _):
        text = os.path.expanduser(self.text.GetValue().strip())
        if os.path.isdir(text):
            self._set_text(text.rstrip(os.sep) + os.sep)
        elif os.path.isfile(text):
            self._accept(text)
        else:   # a single matching name: it is taken
            folder, prefix = self._split()
            names = self._matches(folder, prefix) if os.path.isdir(folder) else []
            if len(names) == 1:
                name, isdir = names[0]
                if isdir:
                    self._set_text(os.path.join(folder, name) + os.sep)
                else:
                    self._accept(os.path.join(folder, name))
            elif not self.spec.style & wx.FD_FILE_MUST_EXIST and text:
                self._accept(text)
            else:
                wx.Bell()

    def on_name_selected(self, event):
        k = event.GetIndex() if hasattr(event, 'GetIndex') else self.names.GetSelection()
        if 0 <= k < len(self._shown):
            name, isdir = self._shown[k]
            if name == '..':   # one click: parent folder
                wx.CallAfter(self.on_parent, None)
                return
            self.text.ChangeValue(os.path.join(self._folder, name) + (os.sep if isdir else ''))

    def on_name_dclick(self, event):
        k = event.GetIndex() if hasattr(event, 'GetIndex') else self.names.GetSelection()
        if 0 <= k < len(self._shown):
            name, isdir = self._shown[k]
            if name == '..':
                return   # (done by the click)
            path = os.path.join(self._folder, name)
            if isdir:
                self._set_text(path + os.sep)
                self.text.SetFocus()
            else:
                self._accept(path)

    def on_recent_dclick(self, _):
        path = self.recent.GetStringSelection()
        if path:
            self._accept(path)

    def _refresh_bookmarks(self, select=None):
        self._bookmarks = read_bookmarks()
        names = [os.path.basename(b) or b for b in self._bookmarks]
        # folder name, or full path if several bookmarks have the same folder name (e.g. scan0001)
        self.bookmarks.Set([b if names.count(n) > 1 else n for n, b in zip(names, self._bookmarks)])
        if select in self._bookmarks:
            self.bookmarks.SetSelection(self._bookmarks.index(select))

    def on_bookmark(self, _):
        k = self.bookmarks.GetSelection()
        if 0 <= k < len(self._bookmarks):
            folder = self._bookmarks[k]
            self.bookmarks.SetToolTip(folder)
            if os.path.isdir(folder):
                self._set_text(folder.rstrip(os.sep) + os.sep)
                self.text.SetFocus()
            else:
                self.info.SetLabel(f'bookmarked folder not found: {folder} (Remove bookmark to delete it)')

    def on_add_bookmark(self, _):
        folder, _ = self._split()
        if not os.path.isdir(folder):
            wx.Bell()
            return
        folder = os.path.abspath(folder).rstrip(os.sep) or os.sep
        add_bookmark(folder)
        self._refresh_bookmarks(select=folder)
        self.info.SetLabel(f'bookmark added: {folder}')

    def on_remove_bookmark(self, _):
        k = self.bookmarks.GetSelection()
        if not 0 <= k < len(self._bookmarks):
            self.info.SetLabel('select the bookmark to remove in the list of bookmarked folders')
            wx.Bell()
            return
        folder = self._bookmarks[k]
        remove_bookmark(folder)
        self._refresh_bookmarks()
        self.info.SetLabel(f'bookmark removed: {folder}')

    def on_parent(self, _):
        folder, _ = self._split()
        self._set_text(os.path.dirname(os.path.abspath(folder).rstrip(os.sep)) + os.sep)
        self.text.SetFocus()

    def on_browse(self, _):
        folder, prefix = self._split()
        dlg = wx.FileDialog(self, self.spec.message, folder if os.path.isdir(folder) else '', prefix,
                            self.spec.wildcard, self.spec.style)
        dlg.SetFilterIndex(self.filter.GetSelection())
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            dlg.Destroy()
            self._accept(path)
            return
        dlg.Destroy()
