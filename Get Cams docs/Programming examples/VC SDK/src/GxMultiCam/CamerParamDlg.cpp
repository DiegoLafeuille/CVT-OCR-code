// CamerParamDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxMultiCam.h"
#include "CamerParamDlg.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CCamerParamDlg dialog


CCamerParamDlg::CCamerParamDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CCamerParamDlg::IDD, pParent)
	, m_bAutoWhiteBalance(false)
	, m_bSpeedLevel(false)
	, m_nAutoWhiteBalance(0)
{
	//{{AFX_DATA_INIT(CCamerParamDlg)
	m_nSpeedLevel = 0;
	m_dShutterValue = 0.0;
	m_dGainValue = 0.0;
	//}}AFX_DATA_INIT
 
	memset(&m_stGainFloatRange,0,sizeof(GX_FLOAT_RANGE));
	memset(&m_stShutterFloatRange,0,sizeof(GX_FLOAT_RANGE));

	m_pMainFrame = (CMainFrame*)AfxGetMainWnd();
}


void CCamerParamDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CCamerParamDlg)
	DDX_Text(pDX, IDC_EDIT_ACQ_SPEEDLEVEL, m_nSpeedLevel);
	DDX_Text(pDX, IDC_EDIT_SHUTTER, m_dShutterValue);
    DDX_Text(pDX, IDC_EDIT_GAIN, m_dGainValue);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CCamerParamDlg, CDialog)
	//{{AFX_MSG_MAP(CCamerParamDlg)
	ON_WM_CLOSE()
	ON_CBN_SELCHANGE(IDC_COMBO_AWB, OnSelchangeComboWriteBlance)
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER, OnKillfocusEditShutter)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN, OnKillfocusEditGain)
	ON_WM_HSCROLL()
	//}}AFX_MSG_MAP
	ON_WM_TIMER()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCamerParamDlg message handlers

BOOL CCamerParamDlg::OnInitDialog() 
{
	CDialog::OnInitDialog();
	
	// TODO: Add extra initialization here
	CString strTemp = _T("");
	strTemp.Format("%sDevice parameters", m_pMainFrame->m_pBaseinfo[m_pMainFrame->m_nOperateID].szDisplayName);
	SetWindowText(strTemp);

	InitUI();

	UpDataUI();
	
	SetTimer(0, 1000, NULL);

	return TRUE;  // return TRUE unless you set the focus to a control
	              // EXCEPTION: OCX Property Pages should return FALSE
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI

\return  void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::InitUI()
{
	GX_DEV_HANDLE hDevice = m_pMainFrame->m_hDevices[m_pMainFrame->m_nOperateID];
	if (hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	int       nValue   = 0;
	
 	InitAutoWhiteBalanceUI(hDevice);
  
	InitShutterUI(hDevice);

	InitGainUI(hDevice);

    InitSpeedLevel(hDevice);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the SpeedLevel for acquiring image
\param   hDevice  [in]  
 
\return  void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::InitSpeedLevel(GX_DEV_HANDLE hDevice)
{
	// Get the pointers for the function 
	CSliderCtrl *pSliderCtrl     = (CSliderCtrl *)GetDlgItem(IDC_SLIDER_ACQ_SPEEDLEVEL);
	CStatic     *pStcSpeedLevel  = (CStatic *)GetDlgItem(IDC_STATIC_ACQ_SPEEDLEVEL);
	CEdit       *pEditSpeedLevel = (CEdit *)GetDlgItem(IDC_EDIT_ACQ_SPEEDLEVEL);

	// Check whether the pointers are available 
	if ((pSliderCtrl == NULL) || (pStcSpeedLevel == NULL) || (pEditSpeedLevel == NULL))
	{
		return;
	}
    
	GX_STATUS emStatus    = GX_STATUS_ERROR;
	int32_t   nStep       = 0;
	CString   strRange    = "";
	int64_t   nSpeedLevel = 0;
	GX_INT_RANGE stIntRange;

	// Check whether the current camera supports AcquisitionSpeedLevel feature
	emStatus = GXIsImplemented(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, &m_bSpeedLevel);
	if (!m_bSpeedLevel)
	{
		pSliderCtrl->EnableWindow(FALSE);
		return;
	}

	// Get the range of AcquisitionSpeedLevel
	emStatus = GXGetIntRange(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, &stIntRange);
	GX_VERIFY(emStatus);
	pSliderCtrl->SetRange((int)stIntRange.nMin, (int)stIntRange.nMax, true);

	// Show the range of AcquisitionSpeedLevel
    strRange.Format("AcquisitionSpeedLevel(%d~%d)", (int)stIntRange.nMin, (int)stIntRange.nMax);
	pStcSpeedLevel->SetWindowText(strRange);
	
	// Get current value of AcquisitionSpeedLevel and update to the UI
	emStatus = GXGetInt(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, &nSpeedLevel);
	GX_VERIFY(emStatus);
	pSliderCtrl->SetPos((int)nSpeedLevel);
	m_nSpeedLevel = (int)nSpeedLevel;
   
	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the Gain.
\param   hDevice  [in]  
 
\return  void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::InitGainUI(GX_DEV_HANDLE hDevice)
{
	CStatic     *pStcGain     = (CStatic *)GetDlgItem(IDC_STC_GAIN);
	CEdit       *pEditGainVal = (CEdit *)GetDlgItem(IDC_EDIT_GAIN);

	if ((pStcGain == NULL) || (pEditGainVal == NULL))
	{
		return;
	}
    
	GX_STATUS emStatus = GX_STATUS_ERROR;
	CString   strTemp  = "";
	double    dGainVal = 0;

	// Get the range of Gain
	emStatus = GXGetFloatRange(hDevice, GX_FLOAT_GAIN, &m_stGainFloatRange);
	GX_VERIFY(emStatus);

	// Show the range in the static control
    strTemp.Format("Gain(%.4f~%.4f)%s",m_stGainFloatRange.dMin, m_stGainFloatRange.dMax, m_stGainFloatRange.szUnit);
	pStcGain->SetWindowText(strTemp);
	
	// Get current value of Gain and update to the UI
	emStatus = GXGetFloat(hDevice, GX_FLOAT_GAIN, &dGainVal);
	GX_VERIFY(emStatus);

	m_dGainValue = dGainVal;

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ExposureTime.
\param   hDevice  [in]   

\return  void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::InitShutterUI(GX_DEV_HANDLE hDevice)
{
	CStatic     *pStcShutterShow = (CStatic *)GetDlgItem(IDC_STC_SHUTTER_SHOW);
	CEdit       *pEditShutterVal = (CEdit *)GetDlgItem(IDC_EDIT_SHUTTER);

	// Check whether the edit control is available
	if ((pEditShutterVal == NULL) || (pStcShutterShow == NULL))
	{
		return;
	}

	GX_STATUS      emStatus = GX_STATUS_ERROR;
	CString        strTemp  = "";
	double         dShutterValue   = 0.0;

	// Get the range of ExposureTime
	emStatus = GXGetFloatRange(hDevice, GX_FLOAT_EXPOSURE_TIME, &m_stShutterFloatRange);
	GX_VERIFY(emStatus);


	strTemp.Format("ExposureTime(%.4f~%.4f)%s", m_stShutterFloatRange.dMin, m_stShutterFloatRange.dMax, m_stShutterFloatRange.szUnit);
	pStcShutterShow->SetWindowText(strTemp);

	// Get current value of ExposureTime and update to the UI
	emStatus = GXGetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, &dShutterValue);
	GX_VERIFY(emStatus);

	m_dShutterValue = dShutterValue;

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI of ComboBox
\param   hDevice    [in]   

\return  void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::InitAutoWhiteBalanceUI(GX_DEV_HANDLE hDevice)
{
	CComboBox *pComboBox = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);
	if ((pComboBox == NULL) || (hDevice == NULL))
	{
		return;
	}

	GX_ENUM_DESCRIPTION *pEnum = NULL;
	GX_STATUS   emStatus       = GX_STATUS_ERROR;
	size_t      nbufferSize    = 0;
	int64_t     nEnumValue     = 0;
	uint32_t    nEntryNum      = 0;
	bool        bImplemented   = false;
	int         nCursel        = 0;

	//Check whether the device supports BalanceWhiteAuto
	emStatus = GXIsImplemented(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &bImplemented);
	GX_VERIFY(emStatus);
	m_bAutoWhiteBalance = bImplemented;
	if (!bImplemented)
	{
		return;
	}

	// Get the enumeration entry the device supports.
	emStatus = GXGetEnumEntryNums(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &nEntryNum);
    GX_VERIFY(emStatus);

	// Allocate memory for getting the description of the enumeration entry
	nbufferSize = nEntryNum * sizeof(GX_ENUM_DESCRIPTION);
	pEnum = new GX_ENUM_DESCRIPTION[nEntryNum];
	if (pEnum == NULL)
	{
		MessageBox("Fail to allocate memory!");
		return;
	}

	emStatus = GXGetEnumDescription(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, pEnum, &nbufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}
	}
    GX_VERIFY(emStatus);

	// Get the enumeration value and show it to the UI 
	emStatus = GXGetEnum(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &nEnumValue);
	GX_VERIFY(emStatus);
	m_nAutoWhiteBalance = nEnumValue;

	// Initialize the options of the ComboBox
	pComboBox->ResetContent();
	for (uint32_t i = 0; i < nEntryNum; i++)
	{
		pComboBox->SetItemData(pComboBox->AddString(pEnum[i].szSymbolic), (uint32_t)pEnum[i].nValue);
		if (pEnum[i].nValue == nEnumValue)
		{
			nCursel = i;
		}
	}

	// Show the value to the UI
	pComboBox->SetCurSel(nCursel);
	
	// Release memory 
	if (pEnum != NULL)
	{  
		delete []pEnum;
		pEnum = NULL;
    }
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in] 

\return void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::ShowErrorString(GX_STATUS emErrorStatus)
{
	char*     pchErrorInfo = NULL;
	size_t    nSize     = 0;
	GX_STATUS emStatus = GX_STATUS_ERROR;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (NULL == pchErrorInfo)
	{
		return;
	}

	// Get error information and show them to user 
	emStatus = GXGetLastError (&emErrorStatus, pchErrorInfo, &nSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		MessageBox("Fail to call GXGetLastError interface!");
	}
	else
	{
		MessageBox((LPCTSTR)pchErrorInfo);
	}
	
	// Release memory 
	if (NULL != pchErrorInfo)
	{
		delete[] pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

void CCamerParamDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	
	KillTimer(0);

	CDialog::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to use the slider to adjust AcquisitionSpeedLevel

  \return void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar) 
{
	// TODO: Add your message handler code here and/or call default
	
	GX_DEV_HANDLE hDevice = m_pMainFrame->m_hDevices[m_pMainFrame->m_nOperateID];
	if (hDevice == NULL)
	{
		return;
	}
	
	GX_STATUS   emStatus = GX_STATUS_ERROR;
	int64_t     nValue = 0;
	double      dValue = 0;
	CSliderCtrl *pSlider = (CSliderCtrl *)pScrollBar;
	
	// Get the current position of the slider
	nValue = pSlider->GetPos();
	
	switch(pScrollBar->GetDlgCtrlID())
	{
	case IDC_SLIDER_ACQ_SPEEDLEVEL: // Adjust AcquisitionSpeedLevel   
		emStatus = GXSetInt(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, nValue);
		m_nSpeedLevel = (int)nValue;
		break;
	default:
		break;
	}	
	
	GX_VERIFY(emStatus);
	UpdateData(FALSE);
	
	CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to set BalanceWhiteAuto

\return void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::OnSelchangeComboWriteBlance() 
{
	// TODO: Add your control notification handler code here
	GX_DEV_HANDLE hDevice = m_pMainFrame->m_hDevices[m_pMainFrame->m_nOperateID];
	if (hDevice == NULL)
	{
		return;
	}

	CComboBox    *pBox     = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);
	GX_STATUS     emStatus = GX_STATUS_SUCCESS;
	int64_t       nEnumVal = 0;
	int           nIndex   = 0;
	
	nIndex    = pBox->GetCurSel();                    // Get current selection
	nEnumVal  = (int64_t)pBox->GetItemData(nIndex);   // Get the enumeration entry of current selection

	//Set BalanceWhiteAuto
	emStatus = GXSetEnum(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, nEnumVal);
	GX_VERIFY(emStatus);

	m_nAutoWhiteBalance = nEnumVal;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to use the control to update UI

\return void
*/
//----------------------------------------------------------------------------------
void CCamerParamDlg::UpDataUI()
{
	BOOL bIsDevOpened   = m_pMainFrame->m_pstCam[m_pMainFrame->m_nOperateID].bIsOpen;
	BOOL bIsSnap        = m_pMainFrame->m_pstCam[m_pMainFrame->m_nOperateID].bIsSnap;

	GetDlgItem(IDC_SLIDER_ACQ_SPEEDLEVEL)->EnableWindow(bIsDevOpened && !bIsSnap && m_bSpeedLevel);
	GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(bIsDevOpened);
	GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(bIsDevOpened);
	GetDlgItem(IDC_COMBO_AWB)->EnableWindow(bIsDevOpened && m_bAutoWhiteBalance);
}
//--------------------------------------------------------------------
/**
\brief    This function demonstrates how to set the ExposureTime

\return   void
*/
//--------------------------------------------------------------------
void CCamerParamDlg::OnKillfocusEditShutter() 
{
	// TODO: Add your control notification handler code here
	UpdateData();

	GX_STATUS status = GX_STATUS_SUCCESS;

	GX_DEV_HANDLE hDevice = m_pMainFrame->m_hDevices[m_pMainFrame->m_nOperateID];

	if (m_dShutterValue > m_stShutterFloatRange.dMax)
	{
		m_dShutterValue = m_stShutterFloatRange.dMax;
	}

	if (m_dShutterValue < m_stShutterFloatRange.dMin)
	{
		m_dShutterValue = m_stShutterFloatRange.dMin;
	}

	status = GXSetFloat(hDevice,GX_FLOAT_EXPOSURE_TIME,m_dShutterValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//---------------------------------------------------------
/**
\brief  This function demonstrates how to set the Gain

\return void
*/
//---------------------------------------------------------
void CCamerParamDlg::OnKillfocusEditGain() 
{
	// TODO: Add your control notification handler code here
	UpdateData();
	
	GX_STATUS status = GX_STATUS_SUCCESS;
	
	GX_DEV_HANDLE hDevice = m_pMainFrame->m_hDevices[m_pMainFrame->m_nOperateID];

	if (m_dGainValue > m_stGainFloatRange.dMax)
	{
		m_dGainValue = m_stGainFloatRange.dMax;
	}

	if (m_dGainValue < m_stGainFloatRange.dMin)
	{
		m_dGainValue = m_stGainFloatRange.dMin;
	}
	
	status = GXSetFloat(hDevice,GX_FLOAT_GAIN,m_dGainValue);
	GX_VERIFY(status);
	
	UpdateData(FALSE);
}

//----------------------------------------------------------------
/**
\brief  This function demonstrates how to deal with message of "Enter" 
\param  pMsg  

\return Success: TRUE Fail:FALSE
*/
//----------------------------------------------------------------
BOOL CCamerParamDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	CWnd *pWnd = NULL;
	int nCtrlID = 0;        

	if (WM_KEYDOWN == pMsg->message && VK_RETURN == pMsg->wParam)
	{
		pWnd = GetFocus();

		nCtrlID = pWnd->GetDlgCtrlID();

		switch(nCtrlID)
		{
		case IDC_EDIT_SHUTTER:
		case IDC_EDIT_GAIN:

			SetFocus();

			break;
		default:
			break;
		}

		return TRUE;
	}
	
	return CDialog::PreTranslateMessage(pMsg);
}

//----------------------------------------------------------------
/**
\brief  This is a callback function of timer for updating UI.
\param  OnTimer ID

\return void
*/
//----------------------------------------------------------------
void CCamerParamDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default

	GX_STATUS status = GX_STATUS_SUCCESS;

	//Set the BalanceWhiteAuto to Once and update to UI
	if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_ONCE)
	{
		int64_t    nWBEnumVal = 0;   //< Save the enumeration entry of BalanceWhiteAuto 
		CComboBox *pAutoWBBox = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);
		GX_DEV_HANDLE hDevice = m_pMainFrame->m_hDevices[m_pMainFrame->m_nOperateID];

		//Get the enumeration entry of BalanceWhiteAuto 
		status = GXGetEnum(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &nWBEnumVal);
		if (status != GX_STATUS_SUCCESS)
		{
			return;
		}

		m_nAutoWhiteBalance = nWBEnumVal;

		//Check whether the the BalanceWhiteAuto has turned to OFF 
		if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_OFF)
		{
			for (int i = 0; i < WHITE_BALANCE_RNTRY; i++)
			{
				if ((int64_t)pAutoWBBox->GetItemData(i) == GX_BALANCE_WHITE_AUTO_OFF)
				{
					// Set BalanceWhiteAuto to OFF
					pAutoWBBox->SetCurSel(i);
					break;
				}
			}	
		}
	}

	CDialog::OnTimer(nIDEvent);
}
