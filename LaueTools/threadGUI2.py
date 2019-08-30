"""
Class to have interaction between GUI and a thread

The Frame class has got the GUI as parent. It starts a thread worker based on a function for a long task f(args, kwargs).
This function f must have in its optional keywords arguments the key 'worker' with value a worker set by the class.
In the function f, the worker attributes _want_abort, fctOutputResults  and method callbackfct() can be called

_want_abort :  1   request to stop the thread
fctOutputResults :  output of function f , at the end of f: worker.fctOutputResults must be set to the normal output of f
callbackfct: method to send message from the function to the GUI Frame class

see also example in indexingAnglesLUT.getUBs_and_MatchingRate() and LaueToolsGUI.ManualIndexFrame.Reckon_2pts_new()

"""

import time
from threading import *
import wx

# Button definitions
ID_START = wx.NewId()
ID_STOP = wx.NewId()

# Define notification event for thread completion
EVT_RESULT_ID = wx.NewId()


def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)


class ResultEvent(wx.PyEvent):
    """Simple event to carry arbitrary result data."""

    def __init__(self, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data


# Thread class that executes processing
class WorkerThread(Thread):
    """Worker Thread Class."""

    def __init__(self, notify_window, fctparams):
        """Init Worker Thread Class."""
        self.fctparams = fctparams
        self.fctOutputResults = None
        Thread.__init__(self)
        self._notify_window = notify_window
        self._want_abort = 0
        # This starts the thread running on creation, but you could
        # also make the GUI thread responsible for calling this
        self.start()

    #     def run_old(self):
    #         """Run Worker Thread."""
    #         # This is the code executing in the new thread. Simulation of
    #         # a long process (well, 10s here) as a simple loop - you will
    #         # need to structure your processing so that you periodically
    #         # peek at the abort variable
    #         for i in range(10):
    #             time.sleep(1)
    #             if self._want_abort:
    #                 # Use a result of None to acknowledge the abort (of
    #                 # course you can use whatever you'd like or even
    #                 # a separate event type)
    #                 wx.PostEvent(self._notify_window, ResultEvent(None))
    #                 return
    #         # Here's where the result would be returned (this is an
    #         # example fixed result of the number 10, but it could be
    #         # any Python object)
    #         wx.PostEvent(self._notify_window, ResultEvent(True))

    def callbackfct(self, arg):
        wx.PostEvent(self._notify_window, ResultEvent(arg))

    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread. Simulation of
        # a long process (well, 10s here) as a simple loop - you will
        # need to structure your processing so that you periodically
        # peek at the abort variable
        # INDEX.getUBs_and_MatchingRate #
        fct, args, keyargs = self.fctparams
        keyargs["worker"] = self
        fct(*args, **keyargs)

    #         for i in range(10):
    #             time.sleep(1)
    #             if self._want_abort:
    #                 # Use a result of None to acknowledge the abort (of
    #                 # course you can use whatever you'd like or even
    #                 # a separate event type)
    #                 wx.PostEvent(self._notify_window, ResultEvent(None))
    #                 return
    # Here's where the result would be returned (this is an
    # example fixed result of the number 10, but it could be
    # any Python object)

    def abort(self):
        """abort worker thread."""
        # Method for use by main thread to signal an abort
        self._want_abort = 1


# GUI Frame class that spins off the worker thread
class ThreadHandlingFrame(wx.Frame):
    """simple Frame Class handling a thread (worker) to started and aborted"""

    def __init__(self, parent, _id, threadFunctionParams=None,
                                parentAttributeName_Result=None,
                                parentNextFunction=None):
        """Create the MainFrame.
        @param threadFunctionParams: threadfunction parameters: list of function name, function arguments, function optional argument (dict)
        @param parentAttributeName_Result: string for attribute name of parent to put the result of the thread function in
        @param parentNextFunction: existing parent method to launch after thread completion or abortion
        """
        wx.Frame.__init__(self, parent, _id, "Searching for Orientation Matrix", size=(400, 200))

        self.threadFunctionParams = threadFunctionParams
        self.parentAttributeName_Result = parentAttributeName_Result
        self.parentNextFunction = parentNextFunction
        self.parent = parent

        # Dumb sample frame with two buttons
        self.startbtn = wx.Button(self, ID_START, "Start")
        self.stopbtn = wx.Button(self, ID_STOP, "Stop")
        self.status = wx.StaticText(self, -1, "", pos=(0, 100))

        self.startbtn.SetBackgroundColour("green")
        self.stopbtn.SetBackgroundColour("red")

        self.Bind(wx.EVT_BUTTON, self.OnStart, id=ID_START)
        self.Bind(wx.EVT_BUTTON, self.OnStop, id=ID_STOP)

        self.layout()

        # Set up event handler for any worker thread results
        EVT_RESULT(self, self.OnResult)

        # And indicate we don't have a worker thread yet
        self.worker = None

    def layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.startbtn, 0, wx.ALL | wx.EXPAND)
        vbox.Add(self.stopbtn, 0, wx.ALL | wx.EXPAND)
        vbox.Add(self.status, 0, wx.ALL | wx.EXPAND)
        self.SetSizer(vbox)

    def OnStart(self, _):
        """Start Computation."""
        # Trigger the worker thread unless it's already busy
        if not self.worker:
            self.status.SetLabel("Computation running... Please Wait.")
            self.worker = WorkerThread(self, self.threadFunctionParams)

    def OnStop(self, _):
        """Stop Computation."""
        # Flag the worker thread to stop if running
        if self.worker:
            self.status.SetLabel("Trying to abort computation")
            self.worker.abort()

    def OnResult(self, event):
        """Show Result status."""
        if event.data is None:
            # Thread aborted (using our convention of None return)
            self.status.SetLabel("Computation aborted")
        elif event.data in (True,):
            print("finished by True event.data")

        else:
            print("finished!    triggered by event.data %s" % event.data)
            # Process results here
            self.status.SetLabel("Computation Result: %s" % event.data)

        # In either event, the worker is done
        #         print "self.parent", self.parent
        #         print "self.worker.fctOutputResults", self.worker.fctOutputResults
        setattr(self.parent, self.parentAttributeName_Result, self.worker.fctOutputResults)
        self.parentNextFunction()
        self.worker = None
        #         print "closing two seconds !"
        #         time.sleep(2)
        self.Close()


APP_SIZE_X = 300
APP_SIZE_Y = 200


class testFrame(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, size=(APP_SIZE_X, APP_SIZE_Y))

        wx.Button(self, 1, "Close", (50, 130))
        wx.Button(self, 2, "Compute", (150, 130), (110, -1))

        self.Bind(wx.EVT_BUTTON, self.OnClose, id=1)
        self.Bind(wx.EVT_BUTTON, self.OnCompute, id=2)

        self.Centre()

    def OnClose(self, event):
        if self.TGframe is not None:
            if self.TGframe.worker is not None:
                self.TGframe.worker.abort()

        self.Close(True)

    def HandleResults(self):
        print("Results of 'OnCompute()':", self.MyResults)

    def readdata(self):
        self.data = list(range(200))

    def OnCompute(self, event):

        self.MyResults = None

        self.readdata()
        arg1 = self.data
        arg2 = self.data[5:-5]
        arg3 = 0

        fctparams = [
            myfunction,
            (arg1, arg2, arg3),
            {"keyarg1": "foo", "keyarg2": [1, 2, 3], "keyarg3": (0, -1, "bar")},
        ]

        self.TGframe = ThreadHandlingFrame(
            self,
            -1,
            threadFunctionParams=fctparams,
            parentAttributeName_Result="MyResults",  # attribute where to put the output of function
            parentNextFunction=self.HandleResults,
        )  # function to be called after completion of stop
        self.TGframe.OnStart(1)
        self.TGframe.startbtn.SetFocus()
        self.TGframe.Show(True)
        self.TGframe.Centre()


def myfunction(
    a, b, c, keyarg1=None, keyarg2=1, keyarg3=0, worker=None, otherkeyarg=True
):
    """
    the function must contain the worker keyword arg and use it to communicate with client class
    Two things to do:
    - place  several 'if worker._want_abort'  conditions
    - set worker.fctOutputResults to the data you want to handle after this thread
     (parent.parentAttributeName_Result will be pointed to these data)
    
    """

    WORKEREXISTS = False
    if worker is not None:
        WORKEREXISTS = True

    # -------- core for computations with loop (for or while)
    res = []
    nbelem = len(b)
    print("b", b)
    NORMALRUNNING = True
    for k in range(nbelem):
        val = 2 * a[k] + b[k]
        res.append(val)
        time.sleep(0.3)
        print("val: %d  k=%d/%d" % (val, k, nbelem))

        if (
            WORKEREXISTS and worker._want_abort
        ):  # places in loops where thread can be stopped
            worker.callbackfct("ABORTED")
            NORMALRUNNING = False

            break

    if WORKEREXISTS:
        if NORMALRUNNING:
            worker.callbackfct("COMPLETED!")
        worker.fctOutputResults = res
        # worker attribute to set with results to be send to parent client class
        # (by specific attribute set by user)

    print("res", res)
    return res


if __name__ == "__main__":

    GUIApp = wx.App()
    GUIframe = testFrame(None, -1, "example GUI thread")
    GUIframe.Show()
    GUIApp.MainLoop()
