; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CAboutDlg
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "GxViewer.h"
LastPage=0

ClassCount=5
Class1=CGxViewerApp
Class2=CGxViewerDoc
Class3=CGxViewerView
Class4=CMainFrame

ResourceCount=2
Resource1=IDR_MAINFRAME
Class5=CAboutDlg
Resource2=IDD_ABOUTBOX

[CLS:CGxViewerApp]
Type=0
HeaderFile=GxViewer.h
ImplementationFile=GxViewer.cpp
Filter=N

[CLS:CGxViewerDoc]
Type=0
HeaderFile=GxViewerDoc.h
ImplementationFile=GxViewerDoc.cpp
Filter=N

[CLS:CGxViewerView]
Type=0
HeaderFile=GxViewerView.h
ImplementationFile=GxViewerView.cpp
Filter=C


[CLS:CMainFrame]
Type=0
HeaderFile=MainFrm.h
ImplementationFile=MainFrm.cpp
Filter=T
BaseClass=CFrameWnd
VirtualFilter=fWC
LastObject=ID_CLOSE_DEVICE




[CLS:CAboutDlg]
Type=0
HeaderFile=GxViewer.cpp
ImplementationFile=GxViewer.cpp
Filter=D
BaseClass=CDialog
VirtualFilter=dWC
LastObject=IDC_STATIC_PRODUCTVERSION

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=4
Control1=IDC_STATIC,static,1342177283
Control2=IDC_STATIC_PRODUCTVERSION,static,1342308480
Control3=IDC_STATIC_LEGALCOPYRIGHT,static,1342308352
Control4=IDOK,button,1342373889

[MNU:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_OPEN_DEVICE
Command2=ID_START_SNAP
Command3=ID_STOP_SNAP
Command4=ID_CLOSE_DEVICE
CommandCount=4

[ACL:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_EDIT_UNDO
Command5=ID_EDIT_CUT
Command6=ID_EDIT_COPY
Command7=ID_EDIT_PASTE
Command8=ID_EDIT_UNDO
Command9=ID_EDIT_CUT
Command10=ID_EDIT_COPY
Command11=ID_EDIT_PASTE
Command12=ID_NEXT_PANE
Command13=ID_PREV_PANE
CommandCount=13

