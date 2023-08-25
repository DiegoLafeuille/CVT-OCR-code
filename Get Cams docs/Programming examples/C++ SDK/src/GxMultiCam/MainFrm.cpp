// MainFrm.cpp : implementation of the CMainFrame class
//
#include "stdafx.h"
#include "GxMultiCam.h"
#include "MainFrm.h"
#include "ControlDlg.h"
#include "GxMultiCamDoc.h"
#include "GxMultiCamView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	ON_WM_CREATE()
	ON_WM_CLOSE()
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,          // status line indicator
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};


// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_objControlDlg = NULL;
}

CMainFrame::~CMainFrame()
{
}


int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;
	CMainFrame::SetMenu(NULL);

	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	m_strTitle = "GxMultiCam";
	cs.cx = 1024;
	cs.cy = 700;
	if( !CFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs
	cs.style = cs.style&~WS_MAXIMIZEBOX&~WS_THICKFRAME&~FWS_ADDTOTITLE ; 

	return TRUE;
}


// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}

#endif //_DEBUG


// CMainFrame message handlers
BOOL CMainFrame::OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext)
{
	BOOL bRet = TRUE;
	// Split the window into two columns
	bRet = m_wndSplitter.CreateStatic(this,1,2,WS_CHILD|WS_VISIBLE|WS_BORDER);
	bRet = m_wndSplitterLeft.CreateStatic(&m_wndSplitter,2,2,WS_CHILD|WS_VISIBLE ,m_wndSplitter.IdFromRowCol(0,1));

	// Get dialog client area
	CRect rectView;
	this->GetClientRect(&rectView);
	CSize MuitCamSize(rectView.Width()-rectView.Width()/4-80, rectView.Height());
	CSize MuitCamChildSize(MuitCamSize.cx/2,MuitCamSize.cy/2);
	CSize ControlDlgSize(rectView.Width()/4+80,rectView.Height());

	bRet = m_wndSplitter.CreateView(0,0,RUNTIME_CLASS(CControlDlg),
		ControlDlgSize,
		pContext);
	bRet = m_wndSplitterLeft.CreateView(0,0,RUNTIME_CLASS(CGxMuitCamView),
		MuitCamChildSize,
		pContext);

	bRet = m_wndSplitterLeft.CreateView(0,1,RUNTIME_CLASS(CGxMuitCamView),
		MuitCamChildSize,
		pContext);

	bRet = m_wndSplitterLeft.CreateView(1,0,RUNTIME_CLASS(CGxMuitCamView),
		MuitCamChildSize,
		pContext);

	bRet = m_wndSplitterLeft.CreateView(1,1,RUNTIME_CLASS(CGxMuitCamView),
		MuitCamChildSize,
		pContext);

	return bRet;
}

void CMainFrame::OnClose()
{
	// Close timer
	KillTimer(0);


	CFrameWnd::OnClose();
}


