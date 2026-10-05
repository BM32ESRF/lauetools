r"""
RangeSlider: wxPython horizontal slider with two thumbs selecting an interval [low, high] of float
values (e.g. vmin, vmax of a color scale), with a text field on each side (similar to ipywidgets
FloatRangeSlider).

- drag a thumb to move low or high value
- drag the selected interval (between thumbs) to move both values
- click outside the interval to move the nearest thumb there
- type a value in a text field and press Enter (values out of slider range extend the range)
- double click on the track to select the whole range

An EVT_RANGE_SLIDER event is posted at each change made by the user (GetValues() gives (low, high)).
"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import math

import wx
import wx.lib.newevent

RangeSliderEvent, EVT_RANGE_SLIDER = wx.lib.newevent.NewCommandEvent()


def _isfinite(value):
    try:
        return math.isfinite(value)
    except TypeError:
        return False


class _RangeTrack(wx.Window):
    """track and two thumbs of RangeSlider (drawn)"""

    RADIUS = 7

    def __init__(self, rangeslider, size=(200, 26)):
        wx.Window.__init__(self, rangeslider, -1, size=size, style=wx.WANTS_CHARS)
        self.rangeslider = rangeslider
        self.SetMinSize(size)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.dragging = None  # 'low', 'high' or 'interval'
        self.dragoffset = 0.
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, lambda evt: (self.Refresh(), evt.Skip()))
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.OnCaptureLost)

    # conversion between pixel position and value
    def _span(self):
        width = self.GetClientSize()[0]
        return self.RADIUS + 1, max(1, width - 2 * (self.RADIUS + 1))

    def value_to_x(self, value):
        rs = self.rangeslider
        x0, width = self._span()
        return x0 + (value - rs.minvalue) / (rs.maxvalue - rs.minvalue) * width

    def x_to_value(self, x):
        rs = self.rangeslider
        x0, width = self._span()
        fraction = min(1., max(0., (x - x0) / width))
        return rs.minvalue + fraction * (rs.maxvalue - rs.minvalue)

    def OnPaint(self, _):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
        dc.Clear()
        gc = wx.GraphicsContext.Create(dc)
        if gc is None:
            return
        width, height = self.GetClientSize()
        ymid = height / 2.
        x0, span = self._span()
        xlow, xhigh = self.value_to_x(self.rangeslider.low), self.value_to_x(self.rangeslider.high)
        enabled = self.IsEnabled()
        highlight = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT) if enabled else wx.Colour(160, 160, 160)
        # track
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.SetBrush(wx.Brush(wx.Colour(200, 200, 200)))
        gc.DrawRoundedRectangle(x0, ymid - 2, span, 4, 2)
        # selected interval
        gc.SetBrush(wx.Brush(highlight))
        gc.DrawRectangle(xlow, ymid - 3, max(0., xhigh - xlow), 6)
        # thumbs
        gc.SetPen(wx.Pen(highlight, 2))
        gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255)))
        for x in (xlow, xhigh):
            gc.DrawEllipse(x - self.RADIUS, ymid - self.RADIUS, 2 * self.RADIUS, 2 * self.RADIUS)

    def OnLeftDown(self, evt):
        x = evt.GetX()
        xlow, xhigh = self.value_to_x(self.rangeslider.low), self.value_to_x(self.rangeslider.high)
        dlow, dhigh = abs(x - xlow), abs(x - xhigh)
        if min(dlow, dhigh) <= self.RADIUS + 1:
            # thumbs at the same position: thumb to move depends on side of the click
            if dlow == dhigh:
                self.dragging = 'low' if x < xlow else 'high'
            else:
                self.dragging = 'low' if dlow < dhigh else 'high'
        elif xlow < x < xhigh:
            self.dragging = 'interval'
            self.dragoffset = self.x_to_value(x) - self.rangeslider.low
        else:  # outside interval: nearest thumb jumps to the click position
            self.dragging = 'low' if dlow < dhigh else 'high'
            self._move(x)
        if not self.HasCapture():
            self.CaptureMouse()

    def _move(self, x):
        rs = self.rangeslider
        value = self.x_to_value(x)
        if self.dragging == 'low':
            rs._set(min(value, rs.high), rs.high)
        elif self.dragging == 'high':
            rs._set(rs.low, max(value, rs.low))
        elif self.dragging == 'interval':
            width = rs.high - rs.low
            low = min(max(value - self.dragoffset, rs.minvalue), rs.maxvalue - width)
            rs._set(low, low + width)

    def OnMotion(self, evt):
        if self.dragging and evt.Dragging() and evt.LeftIsDown():
            self._move(evt.GetX())

    def OnLeftUp(self, _):
        self.dragging = None
        if self.HasCapture():
            self.ReleaseMouse()

    def OnCaptureLost(self, _):
        self.dragging = None

    def OnDoubleClick(self, _):
        rs = self.rangeslider
        rs._set(rs.minvalue, rs.maxvalue)


class RangeSlider(wx.Panel):
    """horizontal slider with two thumbs selecting [low, high] in [minvalue, maxvalue],
    with a text field of low value on the left and of high value on the right"""

    def __init__(self, parent, minvalue=0., maxvalue=1., low=None, high=None, fmt="%.6g",
                 textwidth=90, tracksize=(220, 26)):
        wx.Panel.__init__(self, parent, -1)
        self.fmt = fmt
        self.minvalue, self.maxvalue = 0., 1.
        self.low, self.high = 0., 1.
        self.lowctrl = wx.TextCtrl(self, -1, "", size=(textwidth, -1), style=wx.TE_PROCESS_ENTER)
        self.highctrl = wx.TextCtrl(self, -1, "", size=(textwidth, -1), style=wx.TE_PROCESS_ENTER)
        self.track = _RangeTrack(self, size=tracksize)
        for ctrl in (self.lowctrl, self.highctrl):
            ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnTextEnter)
            ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnTextKillFocus)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.lowctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        sizer.Add(self.track, 1, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(self.highctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
        self.SetSizer(sizer)

        tip = ("Drag a thumb to change min or max value, drag the interval to move both,\n"
               "click outside the interval to move the nearest thumb there, double click to select "
               "the whole range.\nValues can be typed in the text fields (press Enter)")
        for widget in (self, self.track, self.lowctrl, self.highctrl):
            widget.SetToolTip(tip)

        self.SetRange(minvalue, maxvalue)
        self.SetValues(minvalue if low is None else low, maxvalue if high is None else high)

    # ---- public API
    def SetRange(self, minvalue, maxvalue):
        """set slider limits (current values are kept, range is extended to include them)"""
        if not (_isfinite(minvalue) and _isfinite(maxvalue)):
            minvalue, maxvalue = 0., 1.
        minvalue, maxvalue = float(min(minvalue, maxvalue)), float(max(minvalue, maxvalue))
        if maxvalue == minvalue:
            maxvalue = minvalue + (abs(minvalue) if minvalue else 1.)
        self.minvalue, self.maxvalue = minvalue, maxvalue
        self._clip_range_to_values()
        self.track.Refresh()

    def GetRange(self):
        return self.minvalue, self.maxvalue

    def SetValues(self, low, high):
        """set low and high values (no event posted)"""
        if not (_isfinite(low) and _isfinite(high)):
            return
        low, high = float(min(low, high)), float(max(low, high))
        self.low, self.high = low, high
        self._clip_range_to_values()
        self._update_texts()
        self.track.Refresh()

    def GetValues(self):
        return self.low, self.high

    def Enable(self, enable=True):
        result = wx.Panel.Enable(self, enable)
        self.track.Refresh()
        return result

    # ---- internals
    def _clip_range_to_values(self):
        """extend slider range to include current values"""
        self.minvalue = min(self.minvalue, self.low)
        self.maxvalue = max(self.maxvalue, self.high)

    def _update_texts(self):
        self.lowctrl.ChangeValue(self.fmt % self.low)
        self.highctrl.ChangeValue(self.fmt % self.high)

    def _set(self, low, high):
        """set values from user action and post EVT_RANGE_SLIDER"""
        if (low, high) == (self.low, self.high):
            return
        self.low, self.high = low, high
        self._update_texts()
        self.track.Refresh()
        self.track.Update()
        wx.PostEvent(self, RangeSliderEvent(self.GetId(), low=low, high=high))

    def _apply_texts(self):
        try:
            low, high = float(self.lowctrl.GetValue()), float(self.highctrl.GetValue())
        except ValueError:
            self._update_texts()  # restore valid values
            return
        if not (_isfinite(low) and _isfinite(high)):
            self._update_texts()
            return
        if low > high:  # keep the edited value, move the other one
            if self.lowctrl.HasFocus() or low != self.low:
                high = low
            else:
                low = high
        self.minvalue, self.maxvalue = min(self.minvalue, low), max(self.maxvalue, high)
        self._set(low, high)
        self._update_texts()

    def OnTextEnter(self, _):
        self._apply_texts()

    def OnTextKillFocus(self, evt):
        self._apply_texts()
        evt.Skip()
