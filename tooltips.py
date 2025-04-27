import tkinter
TOOLTIPS_IN_USE = {}
THEME_BG="#3b3b3b"
THEME_FG="white"
def createToolTipAtGivenPos(id: str, root: tkinter.Tk, message: str, command, event: tkinter.Event, ):
    global TOOLTIPS_IN_USE
    TOOLTIPS_IN_USE[id] = [None, None]
    x=root.winfo_pointerx()
    y=root.winfo_pointery()
    def __actualCreateToolTip():
        global TOOLTIPS_IN_USE
        toolTipLabel = tkinter.Label(root, text=message, background=THEME_BG, foreground=THEME_FG)
        toolTipLabel.place(x=x, y=y)
        toolTipLabel.bind("<Button-1>", command)
        TOOLTIPS_IN_USE[id][0] = toolTipLabel
        return True
    TOOLTIPS_IN_USE[id][1] = root.after(1000, __actualCreateToolTip )
    return True
def deleteToolTip(id:str, root:tkinter.Tk):
    try:
        root.after_cancel(TOOLTIPS_IN_USE[id][1])
        toolTipToDelete = TOOLTIPS_IN_USE[id][0]
        toolTipToDelete.destroy()
        del TOOLTIPS_IN_USE[id]
    except Exception as EXP:
        pass
    return True