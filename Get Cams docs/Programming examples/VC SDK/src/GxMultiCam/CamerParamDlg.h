#if !defined(AFX_CAMERPARAMDLG_H__875A1E64_0402_42AB_A20C_F4E36375A831__INCLUDED_)
#define AFX_CAMERPARAMDLG_H__875A1E64_0402_42AB_A20C_F4E36375A831__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// CamerParamDlg.h : header file
//
#include "GxIAPI.h"
#include "MainFrm.h"

#define  WHITE_BALANCE_RNTRY  3        
/////////////////////////////////////////////////////////////////////////////
// CCamerParamDlg dialog

class CCamerParamDlg : public CDialog
{
// Construction
public:
	CCamerParamDlg(CWnd* pParent = NULL);   // standard constructor

	bool                m_bAutoWhiteBalance;       
	bool                m_bSpeedLevel;             

	int64_t             m_nAutoWhiteBalance;       

	GX_FLOAT_RANGE      m_stShutterFloatRange;     
	GX_FLOAT_RANGE      m_stGainFloatRange;        

	CMainFrame         *m_pMainFrame;             
// Dialog Data
	//{{AFX_DATA(CCamerParamDlg)
	enum { IDD = IDD_DLG_CAMERA_PARAM };
	int		m_nSpeedLevel;
	double	m_dShutterValue;
	double	m_dGainValue;
	//}}AFX_DATA

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCamerParamDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	//}}AFX_VIRTUAL


public:
	
	void ShowErrorString(GX_STATUS emErrorStatus);
	

	void UpDataUI();

	void InitUI();

	void InitAutoWhiteBalanceUI(GX_DEV_HANDLE hDevice);
	
	void InitShutterUI(GX_DEV_HANDLE hDevice);

	void InitGainUI(GX_DEV_HANDLE hDevice);

	void InitSpeedLevel(GX_DEV_HANDLE hDevice);

protected:
	// Generated message map functions
	//{{AFX_MSG(CCamerParamDlg)
	virtual BOOL OnInitDialog();
	virtual void DoDataExchange(CDataExchange* pDX);
	afx_msg void OnClose();
	afx_msg void OnSelchangeComboWriteBlance();
	afx_msg void OnKillfocusEditShutter();
	afx_msg void OnKillfocusEditGain();
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()	
public:
	afx_msg void OnTimer(UINT_PTR nIDEvent);
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CAMERPARAMDLG_H__875A1E64_0402_42AB_A20C_F4E36375A831__INCLUDED_)
