
import sys
import wx

# try:
#     from SpecClient_gevent import SpecCommand
# except ImportError:
#     print('-- warning. spec control software and SpecClient_gevent missing ? (normal if you are not at the beamline)')

try:
    sys.path.append('/users/blissadm/local/bliss.git')
    from bliss.config import static

    config = static.get_config()
    from bliss.common import standard

    session =  config.get('test')
    #session.setup()
except ImportError:
    print('-- warning. BLISS control software is missing ? (normal if you are not at the beamline)')

# --- ---------------  Plot limits board  parameters
class MessageCommand(wx.Dialog):
    """
    Class to command with spec
    """

    def __init__(self, parent, _id, title, sentence=None, speccommand=None, specconnection=None):
        """
        initialize board window
        """
        wx.Dialog.__init__(self, parent, _id, title, size=(400, 250))

        self.parent = parent
        #print("self.parent", self.parent)

        self.speccommand = speccommand

        txt1 = wx.StaticText(self, -1, "%s\n\n%s" % (sentence, self.speccommand))

        acceptbtn = wx.Button(self, -1, "OK")
        # tospecbtn = wx.Button(self, -1, "Send to Spec")
        toblissbtn = wx.Button(self, -1, "Send to Bliss")
        cancelbtn = wx.Button(self, -1, "Cancel")

        acceptbtn.Bind(wx.EVT_BUTTON, self.onAccept)
        cancelbtn.Bind(wx.EVT_BUTTON, self.onCancel)
        # tospecbtn.Bind(wx.EVT_BUTTON, self.onCommandtoSpec)
        toblissbtn.Bind(wx.EVT_BUTTON, self.onCommandtoBliss)

        btnssizer = wx.BoxSizer(wx.HORIZONTAL)
        btnssizer.Add(acceptbtn, 0, wx.ALL)
        btnssizer.Add(cancelbtn, 0, wx.ALL)
        # btnssizer.Add(tospecbtn, 0, wx.ALL)
        btnssizer.Add(toblissbtn, 0, wx.ALL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(txt1)
        vbox.Add(btnssizer)
        self.SetSizer(vbox)

    def onAccept(self, _):

        self.Close()

    def onCancel(self, _):

        # todo save old positions and make inverse mvt
        self.Close()

    # def onCommandtoSpec(self, _):

    #     myspec = SpecCommand.SpecCommand("", "crg1:laue")

    #     print("Sending command : " + self.speccommand)

    #     myspec.executeCommand(self.speccommand)

    #     self.Close()
        
    def onCommandtoBliss(self, _):
        
        
        m0=session.env_dict['m0']
        m1=session.env_dict['m1']
        print('toto--- in onCommandtoBliss')
        print('dir(m0)', dir(m0))
        print('dir(session)', dir(session))
        print('\n\ntoto--- in onCommandtoBliss  retour')
        
        standard.mv(m0, 123)
        # standard.mv(m0, 987)

        print("\n\n*******************\n moved motors !!! \n*****************\n\n")

        self.Close()
        