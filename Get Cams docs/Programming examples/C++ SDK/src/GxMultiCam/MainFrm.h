// MainFrm.h : interface of the CMainFrame class
//

#pragma once
class CControlDlg;             ///< Declare the dialog class
class CMainFrame : public CFrameWnd
{
	
	CSplitterWnd m_wndSplitter;          ///< Split the window into left and right
public: // create from serialization only
	CMainFrame();
	DECLARE_DYNCREATE(CMainFrame)

// Attributes
public:

// Operations
public:
	CSplitterWnd m_wndSplitterLeft;      ///< Split the right view into four windows, up, down, left and right
	CControlDlg* m_objControlDlg;

// Overrides
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	BOOL OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext);
	CWnd* m_pWnd[4];

// Implementation
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // control bar embedded members
	CStatusBar  m_wndStatusBar;
	CToolBar    m_wndToolBar;

// Generated message map functions
protected:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnClose();
	DECLARE_MESSAGE_MAP()
};


