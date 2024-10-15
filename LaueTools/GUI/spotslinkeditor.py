# -*- coding: utf-8 -*-
r"""
GUI class to connect spots from two sets
"""
import wx

class LinkEditor(wx.Frame):
    def __init__(self, parent, _id, title, previouslist, millerlist, intensitylist=None):
        # wx.Dialog.__init__(self, parent, id, title, size=(600,500), style=wx.DEFAULT_DIALOG_STYLE)
        #         wx.Dialog.__init__(self, parent, id, title, size=(930, 500), style=wx.OK)

        wx.Frame.__init__(self, parent, _id, title, size=(930, 500))

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox1 = wx.BoxSizer(wx.VERTICAL)
        vbox2 = wx.BoxSizer(wx.VERTICAL)
        vbox3 = wx.GridSizer(2, 2, 0, 0)
        vbox4 = wx.BoxSizer(wx.VERTICAL)
        pnl1 = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)
        pnl2 = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)
        self.lc = wx.ListCtrl(self, -1, style=wx.LC_REPORT)
        self.lc.InsertColumn(0, "Experimental")
        self.lc.InsertColumn(1, "Simulated")
        self.lc.InsertColumn(2, "Miller")
        self.lc.InsertColumn(3, "Intensity")
        self.lc.SetColumnWidth(0, 100)
        self.lc.SetColumnWidth(1, 100)
        self.lc.SetColumnWidth(2, 150)
        self.lc.SetColumnWidth(3, 100)
        vbox1.Add(pnl1, 1, wx.EXPAND | wx.ALL, 3)
        vbox1.Add(pnl2, 1, wx.EXPAND | wx.ALL, 3)
        vbox2.Add(self.lc, 1, wx.EXPAND | wx.ALL, 3)
        self.tc1 = wx.TextCtrl(pnl1, -1)
        self.tc2 = wx.TextCtrl(pnl1, -1)
        vbox3.AddMany([(wx.StaticText(pnl1, -1, "Experimental"), 0, wx.ALIGN_CENTER),
                (self.tc1, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL),
                (wx.StaticText(pnl1, -1, "Simulated"), 0, wx.ALIGN_CENTER_HORIZONTAL),
                (self.tc2, 0)])
        pnl1.SetSizer(vbox3)
        vbox4.Add(wx.Button(pnl2, 10, "Add"), 0, wx.ALIGN_CENTER | wx.TOP, 45)
        vbox4.Add(wx.Button(pnl2, 11, "Remove"), 0, wx.ALIGN_CENTER | wx.TOP, 15)
        vbox4.Add(wx.Button(pnl2, 12, "Clear All"), 0, wx.ALIGN_CENTER | wx.TOP, 15)
        vbox4.Add(wx.Button(pnl2, wx.ID_OK, "Accept and Quit"), 0, wx.ALIGN_CENTER | wx.TOP, 15)
        pnl2.SetSizer(vbox4)
        self.Bind(wx.EVT_BUTTON, self.OnAdd, id=10)
        self.Bind(wx.EVT_BUTTON, self.OnRemove, id=11)
        self.Bind(wx.EVT_BUTTON, self.OnClear, id=12)
        self.Bind(wx.EVT_BUTTON, self.OnClose, id=wx.ID_OK)
        hbox.Add(vbox1, 1, wx.EXPAND)
        hbox.Add(vbox2, 1, wx.EXPAND)
        self.SetSizer(hbox)

        # must be of equal length
        self.millerlist = millerlist
        self.intensitylist = intensitylist

        # list for preparing the returned result
        self.listofpairs = []

        # pre defined links
        self.previouslist = previouslist
        if self.previouslist is not None:
            self.AddList(self.previouslist)

    #         self.ShowModal()

    def AddList(self, prelist_links):
        num_items = self.lc.GetItemCount()
        for elem in prelist_links:
            self.lc.InsertStringItem(num_items, str(int(elem[0])))
            self.lc.SetStringItem(num_items, 1, "unknown")
            # TODO: fill this field but it will change with the simulation parameter
            self.lc.SetStringItem(num_items, 2, "[%d, %d, %d]" % tuple(elem[1:4]))
            if self.intensitylist is not None:
                self.lc.SetStringItem(num_items, 3, str(self.intensitylist[num_items]))
            else:
                self.lc.SetStringItem(num_items, 3, "unknown")
            num_items += 1

    def OnAdd(self, _):
        if not self.tc1.GetValue() or not self.tc2.GetValue():
            return

        num_items = self.lc.GetItemCount()
        self.lc.InsertStringItem(num_items, self.tc1.GetValue())
        self.lc.SetStringItem(num_items, 1, self.tc2.GetValue())
        try:
            self.lc.SetStringItem(num_items, 2, str(self.ReadMiller(int(self.tc2.GetValue()))))
        except IndexError:
            wx.MessageBox("Theoretical spot of index %d does not exist ! \n Please remove it! "
                % int(self.tc2.GetValue()), "INFO")

        if self.intensitylist is not None:
            self.lc.SetStringItem(num_items, 3, str(self.intensitylist[int(self.tc1.GetValue())]))
        else:
            self.lc.SetStringItem(num_items, 3, "unknown")

        self.tc1.Clear()
        self.tc2.Clear()

    def ReadMiller(self, index):
        print("index", index)
        print("self.millerlist[index]", self.millerlist[index])
        return self.millerlist[index]
    def OnRemove(self, _):
        index = self.lc.GetFocusedItem()
        self.lc.DeleteItem(index)

    def OnClose(self, _):
        self.listofpairs = []
        self.linkMiller = []
        self.linkIntensity = []
        for idx in range(self.lc.GetItemCount()):

            item_exp = int(self.lc.GetItem(idx).GetText())

            sim_index = self.lc.GetItem(idx, 1).GetText()
            if sim_index != "unknown":
                item_sim = int(sim_index)
            else:
                item_sim = -1  # integer code for unknown index

            str_miller = str(self.lc.GetItem(idx, 2).GetText())
            print("str_miller", str_miller)
            if "," in str_miller:
                HKL = [float(ind) for ind in str_miller[1:-1].split(",")]
            else:
                HKL = [float(ind) for ind in str_miller[1:-1].split()]
            H, K, L = HKL

            intensity = float(self.lc.GetItem(idx, 3).GetText())

            # print [item_exp,item_sim,HKL]
            self.listofpairs.append([item_exp, item_sim])
            self.linkMiller.append([float(item_exp), H, K, L])
            self.linkIntensity.append(intensity)
            # print "\n\n self.listofpairs in editor class",self.listofpairs
        self.Close()
        return self.listofpairs, self.linkMiller, self.linkIntensity

    def OnClear(self, _):
        self.lc.DeleteAllItems()


if __name__ == "__main__":

    # test data:
    # old links between exp spots and assigned miller indices
    nb_of_exp_spots = 40
    previouslist = [[0, -1, 5, 2],
                        [1, 2, 3, 8],
                        [8, 2, 0, 2],
                        [10, -1, -1, -1],
                        [nb_of_exp_spots - 1, 0, 0, 1]]  # exp spot, h,k,l

    # some miller indices that can be picked up by theoretical spot index
    millerlist = [[-2, 0, 0],
                    [0, 2, 0],
                    [1, 1, 1],
                    [0, 0, 1],
                    [-1, 1, 1],
                    [1, 0, 0],
                    [-3, 1, 3]]

    # intensity list corresponding of exp. spots intensity
    intensitylist = [-6.2, 6.0, 10.0, -0.100, 1000.0, -56.0, 9.25, 236.0]

    class MyApp(wx.App):
        def OnInit(self):
            dia = LinkEditor(None,
                                -1,
                                "Spots Links Editor.py",
                                previouslist,
                                millerlist,
                                intensitylist=intensitylist)
            # dia.ShowModal()
            dia.Destroy()

            print("list of pairs", dia.listofpairs)
            print("list of Miller selection", dia.linkMiller)
            print("list of intensity selection", dia.linkIntensity)
            return True

    app = MyApp(0)
    app.MainLoop()
