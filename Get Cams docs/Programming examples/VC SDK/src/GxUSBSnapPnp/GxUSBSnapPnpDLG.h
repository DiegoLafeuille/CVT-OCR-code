// GxUSBSnapPnpDlg.h : header file
//

#if !defined(AFX_GxUSBSnapPnpDLG_H__B4BF8831_0E24_4FE2_B20F_46AC3BECF299__INCLUDED_)
#define AFX_GxUSBSnapPnpDLG_H__B4BF8831_0E24_4FE2_B20F_46AC3BECF299__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

/////////////////////////////////////////////////////////////////////////////
// CGxUSBSnapPnpDlg dialog
#pragma warning(disable : 4786)
#include "GxIAPI.h"
#include "DxImageProc.h"

//----------------------------------------------------------------------------------
/**
\Class  CGxUSBSnapPnpDlg

\brief  The class of main widow
*/
//----------------------------------------------------------------------------------
class CGxUSBSnapPnpDlg : public CDialog
{
// Construction
public:
	CGxUSBSnapPnpDlg(CWnd* pParent = NULL);

	void InitListHeader();

	/// Register the type of camera and return information 
	BOOL RegHVSnapPnpGUID(HANDLE handle, GUID stGuid);

	void UpdateDeviceList();

// Dialog Data
	//{{AFX_DATA(CGxUSBSnapPnpDlg)
	enum { IDD = IDD_GxUSBSnapPnp_DIALOG };
	CListCtrl	m_listResult;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxUSBSnapPnpDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	virtual LRESULT DefWindowProc(UINT message, WPARAM wParam, LPARAM lParam);
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxUSBSnapPnpDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnClose();
	afx_msg HCURSOR OnQueryDragIcon();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()	
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GxUSBSnapPnpDLG_H__B4BF8831_0E24_4FE2_B20F_46AC3BECF299__INCLUDED_)
